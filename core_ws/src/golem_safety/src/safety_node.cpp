// safety_node: the only writer of the robots' body command topics. SafetyCore makes every
// decision; this node takes the samples from ROS, keeps time on the boot clock, publishes
// what the core returns, and logs the core's events.
//
// Threads:
//   - the relay thread polls /lowstate, the hardware estop and the mode's inputs with
//     take() and calls step() at control.publish_hz, sleeping to absolute deadlines on the
//     boot clock; on exit it calls exit_step() at the same rate until the core is done.
//     It is the only thread that touches SafetyCore, except for estopped();
//   - an executor thread runs the 10 Hz /safety/heartbeat wall timer, which reads only
//     estopped(); the polled subscriptions are in a callback group no executor spins;
//   - an EventLog thread writes every log line, so that stderr can never block the relay;
//   - main() waits for SIGINT, SIGTERM or SIGHUP, which every thread blocks, then has the
//     relay run the exit sequence before rclcpp shuts down.

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "golem_safety/boot_clock.hpp"
#include "golem_safety/chunk_logger.hpp"
#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"
#include "golem_safety/safety_core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

namespace golem_safety
{
namespace
{

using std_msgs::msg::Bool;
using unitree_hg::msg::LowCmd;
using unitree_hg::msg::LowState;

const char * const kHeartbeatTopic = "/safety/heartbeat";
constexpr std::chrono::milliseconds kHeartbeatPeriod{100};  // 10 Hz
// Lines waiting for stderr; beyond this, new lines are dropped and counted.
constexpr size_t kMaxQueuedLines = 10000;
// How long the exit waits for stderr to take the last lines, s.
constexpr double kLogDrainSeconds = 1.0;
// How long an exit sequence may overrun its ramp or burst before it counts as failed, s;
// it covers the short weight-0 hold that follows the arm_sdk ramp (SafetyCore::exit_step).
constexpr double kExitMarginSeconds = 1.0;

std::string json_string(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
      out += escaped;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

const char * signal_name(int signum)
{
  switch (signum) {
    case SIGINT:
      return "SIGINT";
    case SIGTERM:
      return "SIGTERM";
    case SIGHUP:
      return "SIGHUP";
  }
  return "an exit signal";
}

// Log lines written by a thread of their own: post() never blocks on stderr or throws, so
// a closed, hung-up or stalled stderr (a dead ros2 launch, a closed terminal, a reader
// that stopped reading) cannot stop the relay or the hand-back. rclcpp writes console
// output under a global mutex, so once the relay runs nothing else in the node logs
// through rclcpp.
class EventLog
{
public:
  explicit EventLog(rclcpp::Logger logger)
  : logger_(std::move(logger)), thread_([this] {run();}) {}

  // Only on a path that skipped stop(); it waits for stderr to take the queued lines.
  ~EventLog()
  {
    if (thread_.joinable()) {
      stop();
    }
  }

  EventLog(const EventLog &) = delete;
  EventLog & operator=(const EventLog &) = delete;

  void post(Event::Kind kind, std::string text) noexcept
  {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (pending_ >= kMaxQueuedLines) {
        ++dropped_;
        return;
      }
      lines_.push_back(Event{kind, std::move(text)});
      ++pending_;
    } catch (...) {
      return;
    }
    wake_.notify_all();
  }

  // Waits up to `seconds` for every posted line to be written; false if stderr is stalled.
  bool drain(double seconds)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return written_.wait_for(
      lock, std::chrono::duration<double>(seconds), [this] {return pending_ == 0;});
  }

  // Joins the writer thread; only after drain() returned true, or it may wait for ever.
  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }

private:
  void run()
  {
    for (;;) {
      Event line{Event::Info, ""};
      size_t dropped = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] {return !lines_.empty() || stopping_;});
        if (lines_.empty()) {
          return;
        }
        line = std::move(lines_.front());
        lines_.pop_front();
        dropped = dropped_;
        dropped_ = 0;
      }
      if (dropped > 0) {
        write(
          Event{Event::Warn, "{\"count\": " + std::to_string(dropped) +
            ", \"event\": \"log_lines_dropped\"}"});
      }
      write(line);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --pending_;
      }
      written_.notify_all();
    }
  }

  void write(const Event & line) const
  {
    switch (line.kind) {
      case Event::Info:
        RCLCPP_INFO(logger_, "%s", line.text.c_str());
        break;
      case Event::Warn:
        RCLCPP_WARN(logger_, "%s", line.text.c_str());
        break;
      case Event::Error:
        RCLCPP_ERROR(logger_, "%s", line.text.c_str());
        break;
    }
  }

  const rclcpp::Logger logger_;
  std::mutex mutex_;
  std::condition_variable wake_, written_;
  std::deque<Event> lines_;  // mutex_
  size_t pending_ = 0;  // mutex_: posted and not yet written, the one being written included
  size_t dropped_ = 0;  // mutex_
  bool stopping_ = false;  // mutex_
  std::thread thread_;
};

// The relay loop and the exit sequence, on one thread.
class Relay
{
public:
  struct CommandInput
  {
    Input input;
    rclcpp::Subscription<LowCmd>::SharedPtr subscription;
  };

  Relay(
    SafetyCore & core, const SafetyConfig & config, EventLog & log, ChunkLogger * chunks,
    rclcpp::Publisher<LowCmd>::SharedPtr out,
    rclcpp::Subscription<LowState>::SharedPtr state, std::vector<CommandInput> inputs,
    rclcpp::Subscription<Bool>::SharedPtr estop_triggered,
    rclcpp::Subscription<Bool>::SharedPtr estop_plugged_in)
  : core_(core), period_(1.0 / config.control.publish_hz), log_(log), chunks_(chunks),
    out_(std::move(out)), state_(std::move(state)), inputs_(std::move(inputs)),
    estop_triggered_(std::move(estop_triggered)), estop_plugged_in_(std::move(estop_plugged_in))
  {}

  // Only on a path that skipped join(): the exit sequence runs, bounded.
  ~Relay()
  {
    if (thread_.joinable()) {
      request_exit();
      if (!join(kDestructorJoinSeconds)) {
        thread_.detach();
      }
    }
  }

  Relay(const Relay &) = delete;
  Relay & operator=(const Relay &) = delete;

  void start() {thread_ = std::thread([this] {run();});}

  void request_exit() {exit_requested_.store(true, std::memory_order_release);}

  // Whether the thread has returned: after the exit sequence, or on a failure.
  bool finished() const {return finished_.load(std::memory_order_acquire);}
  bool failed() const {return failed_.load(std::memory_order_acquire);}

  // Waits up to `seconds` for the thread to return, then joins it; false if it has not.
  bool join(double seconds)
  {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!done_.wait_for(
          lock, std::chrono::duration<double>(seconds), [this] {return finished();}))
      {
        return false;
      }
    }
    thread_.join();
    return true;
  }

private:
  static constexpr double kDestructorJoinSeconds = 5.0;

  void run() noexcept
  {
    try {
      loop();
    } catch (const std::exception & e) {
      failed_.store(true, std::memory_order_release);
      log_.post(Event::Error, std::string("the relay loop failed: ") + e.what());
    } catch (...) {
      failed_.store(true, std::memory_order_release);
      log_.post(Event::Error, "the relay loop failed");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_.store(true, std::memory_order_release);
    }
    done_.notify_all();
  }

  void loop()
  {
    double next = boot_time();
    bool exiting = false;
    for (;;) {
      exiting = exiting || exit_requested_.load(std::memory_order_acquire);
      const double now = boot_time();
      take_state(now);
      take_estop(now);
      if (!exiting) {
        take_commands(now);
      }
      const std::optional<LowCmd> out = exiting ? core_.exit_step(now) : core_.step(now);
      if (out) {
        out_->publish(*out);
      }
      // after the command is out: the estop report is built here
      for (Event & event : core_.drain_events()) {
        log_.post(event.kind, std::move(event.text));
      }
      report_dropped_samples();
      if (exiting && !out) {
        return;
      }
      next += period_;
      // a cycle that ran more than a period late restarts the schedule: a slow cycle lowers
      // the rate instead of bursting to catch up
      const double after = boot_time();
      if (next < after - period_) {
        next = after;
      }
      sleep_until_boot_time(next);
    }
  }

  void take_state(double now)
  {
    if (!state_->take(state_msg_, info_)) {
      return;
    }
    core_.on_state(state_msg_, now);
    if (chunks_ != nullptr) {
      if (std::optional<LogSample> sample = core_.take_log_sample()) {
        chunks_->log(std::move(*sample));
      }
    }
  }

  // A hardware estop sample is a value on each topic: on_estop runs once both have one,
  // and then with the newest pair on every new value of either.
  void take_estop(double now)
  {
    if (!estop_triggered_) {
      return;
    }
    bool fresh = false;
    if (estop_triggered_->take(bool_msg_, info_)) {
      triggered_ = bool_msg_.data;
      fresh = true;
    }
    if (estop_plugged_in_->take(bool_msg_, info_)) {
      plugged_in_ = bool_msg_.data;
      fresh = true;
    }
    if (fresh && triggered_ && plugged_in_) {
      core_.on_estop(*triggered_, *plugged_in_, now);
    }
  }

  void take_commands(double now)
  {
    for (const CommandInput & in : inputs_) {
      if (!in.subscription->take(cmd_msg_, info_)) {
        continue;
      }
      const rmw_gid_t & gid = info_.get_rmw_message_info().publisher_gid;
      core_.on_command(
        in.input, cmd_msg_,
        WriterId(reinterpret_cast<const char *>(gid.data), sizeof(gid.data)), now);
    }
  }

  // Samples the chunk logger refused (another motor count, or a full queue), once.
  void report_dropped_samples()
  {
    if (chunks_ == nullptr || dropped_reported_ || chunks_->samples_dropped() == 0) {
      return;
    }
    dropped_reported_ = true;
    log_.post(
      Event::Warn, "the chunk logger is dropping samples: " +
      std::to_string(chunks_->samples_dropped()) + " so far (the total is logged at exit)");
  }

  SafetyCore & core_;
  const double period_;  // s
  EventLog & log_;
  ChunkLogger * const chunks_;
  const rclcpp::Publisher<LowCmd>::SharedPtr out_;
  const rclcpp::Subscription<LowState>::SharedPtr state_;
  const std::vector<CommandInput> inputs_;
  const rclcpp::Subscription<Bool>::SharedPtr estop_triggered_, estop_plugged_in_;

  // the relay thread's
  LowState state_msg_;
  LowCmd cmd_msg_;
  Bool bool_msg_;
  rclcpp::MessageInfo info_;
  std::optional<bool> triggered_, plugged_in_;
  bool dropped_reported_ = false;

  std::atomic<bool> exit_requested_{false};
  std::atomic<bool> finished_{false};
  std::atomic<bool> failed_{false};
  std::mutex mutex_;
  std::condition_variable done_;
  std::thread thread_;
};

// The node's parameters, read once at start.
struct Parameters
{
  std::string robot;
  std::string config;
  bool sim = false;
  bool use_estop = false;
};

Parameters read_parameters(rclcpp::Node & node)
{
  Parameters p;
  p.robot = node.declare_parameter<std::string>("robot", "");
  p.config = node.declare_parameter<std::string>("config", "");
  p.sim = node.declare_parameter<bool>("sim", false);
  p.use_estop = node.declare_parameter<bool>("use_estop", false);
  if (p.robot.empty()) {
    throw std::runtime_error("parameter 'robot' is required: h12 or g1");
  }
  if (p.config.empty()) {
    throw std::runtime_error(
            "parameter 'config' is required: a config name under "
            "share/golem_safety/config/<robot>/, or a path");
  }
  return p;
}

// Logs the reason on one line, at once (nothing is running yet), and returns the exit code.
int refuse(const rclcpp::Logger & logger, std::string reason)
{
  for (char & c : reason) {
    if (c == '\n') {
      c = ' ';
    }
  }
  RCLCPP_FATAL(logger, "safety_node refuses to start: %s", reason.c_str());
  return 1;
}

std::vector<Relay::CommandInput> command_inputs(
  rclcpp::Node & node, const SafetyConfig & config, const rclcpp::QoS & qos,
  const rclcpp::SubscriptionOptions & options)
{
  auto subscribe = [&](Input input, const std::string & topic) {
      return Relay::CommandInput{
        input, node.create_subscription<LowCmd>(
          topic, qos, [](LowCmd::ConstSharedPtr) {}, options)};
    };
  switch (config.mode) {
    case Mode::FullBody:
      return {subscribe(Input::Full, config.topics.low_cmd_in)};
    case Mode::Split:
      return {subscribe(Input::Lower, config.topics.low_cmd_lower_in),
        subscribe(Input::Upper, config.topics.low_cmd_upper_in)};
    case Mode::ArmSdk:
      return {subscribe(Input::Upper, config.topics.low_cmd_upper_in)};
  }
  return {};
}

// The longest the exit sequence's ramp or burst takes, s: the arm_sdk weight ramp, or the
// rest of an estop burst.
double exit_seconds(const SafetyConfig & config)
{
  if (config.mode == Mode::ArmSdk) {
    return config.arm_sdk.release_seconds;
  }
  return config.estop_response.burst_count / config.control.publish_hz;
}

int run(int argc, char ** argv, const sigset_t & exit_signals)
{
  try {
    rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "safety_node refuses to start: %s\n", e.what());
    return 1;
  }
  auto node = std::make_shared<rclcpp::Node>("safety_node");
  const rclcpp::Logger logger = node->get_logger();

  Parameters params;
  std::unique_ptr<RobotProfile> profile;
  std::unique_ptr<SafetyConfig> config;
  std::unique_ptr<SafetyCore> core;
  std::unique_ptr<ChunkLogger> chunks;
  size_t domain = 0;
  try {
    params = read_parameters(*node);
    // before any publisher or subscription of the node's own
    domain = node->get_node_base_interface()->get_context()->get_domain_id();
    if (params.sim && domain == 0) {
      const int code = refuse(
        logger, "sim:=true on ROS domain 0, the real robots' command bus: set ROS_DOMAIN_ID "
        "to the simulator's domain");
      rclcpp::shutdown();
      return code;
    }
    profile = std::make_unique<RobotProfile>(load_robot_profile(params.robot));
    config = std::make_unique<SafetyConfig>(load_safety_config(params.config, *profile));
    core = std::make_unique<SafetyCore>(
      *profile, *config, params.use_estop, static_cast<int>(domain));
    if (config->logging.enabled) {
      chunks = std::make_unique<ChunkLogger>(
        run_log_directory(config->logging.base_dir), config->logging.chunk_prefix,
        ChunkLogger::kDefaultSamplesPerChunk);
    }
  } catch (const std::exception & e) {
    const std::string robot = params.robot.empty() ? std::string("?") : params.robot;
    const std::string name = params.config.empty() ? std::string("?") : params.config;
    const int code = refuse(logger, "robot " + robot + ", config " + name + ": " + e.what());
    rclcpp::shutdown();
    return code;
  }

  // Newest only: each input and /lowstate keeps one sample, taken by the relay thread.
  // Best effort matches the Unitree SDK's writers as well as ROS ones.
  const rclcpp::QoS newest = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
  // The polled subscriptions' group is never added to an executor, so their callbacks
  // never run.
  rclcpp::SubscriptionOptions polled;
  polled.callback_group =
    node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
  // Reliable keep-last 1, the Unitree SDK's writer: it matches best-effort and reliable
  // readers, the robots' and the sims' included.
  auto out = node->create_publisher<LowCmd>(
    config->topics.low_cmd_out, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
  auto state = node->create_subscription<LowState>(
    config->topics.low_state, newest, [](LowState::ConstSharedPtr) {}, polled);
  std::vector<Relay::CommandInput> inputs = command_inputs(*node, *config, newest, polled);
  rclcpp::Subscription<Bool>::SharedPtr estop_triggered, estop_plugged_in;
  if (params.use_estop) {
    estop_triggered = node->create_subscription<Bool>(
      config->estop.triggered_topic, newest, [](Bool::ConstSharedPtr) {}, polled);
    estop_plugged_in = node->create_subscription<Bool>(
      config->estop.plugged_in_topic, newest, [](Bool::ConstSharedPtr) {}, polled);
  }

  // true until the estop latches; on the steady clock, whatever use_sim_time says
  auto heartbeat = node->create_publisher<Bool>(kHeartbeatTopic, rclcpp::QoS(10));
  SafetyCore & safety = *core;
  auto heartbeat_timer = node->create_wall_timer(
    kHeartbeatPeriod, [&safety, &heartbeat] {
      Bool msg;
      msg.data = !safety.estopped();
      heartbeat->publish(msg);
    });

  EventLog log(logger);
  Relay relay(
    *core, *config, log, chunks.get(), out, state, std::move(inputs), estop_triggered,
    estop_plugged_in);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor] {executor.spin();});
  relay.start();
  {
    std::string started = "{\"config\": " + json_string(config->source) +
      ", \"event\": \"relay_started\"";
    if (chunks) {
      started += ", \"log_dir\": " + json_string(chunks->directory());
    }
    started += ", \"mode\": " + json_string(mode_name(config->mode)) + ", \"robot\": " +
      json_string(profile->robot) + ", \"sim\": " + (params.sim ? "true" : "false") +
      ", \"use_estop\": " + (params.use_estop ? "true" : "false") + "}";
    log.post(Event::Info, started);
  }

  // Until an exit signal, or the relay ending by itself, which is a failure.
  int signum = 0;
  while (!relay.finished()) {
    const timespec poll{0, 100000000};  // 0.1 s
    siginfo_t info;
    const int taken = sigtimedwait(&exit_signals, &info, &poll);
    if (taken > 0) {
      signum = taken;
      break;
    }
  }
  int code = 0;
  bool joined = true;
  if (signum != 0) {
    log.post(
      Event::Info, "{\"event\": \"exit_requested\", \"reason\": " +
      json_string(std::string("Exit requested (") + signal_name(signum) + ")") + "}");
    relay.request_exit();
    const double budget = exit_seconds(*config) + kExitMarginSeconds;
    joined = relay.join(budget);
    if (!joined) {
      log.post(
        Event::Error, "{\"event\": \"exit_hand_back_timeout\", \"seconds\": " +
        std::to_string(budget) + "}");
      code = 1;
    }
  } else {
    joined = relay.join(0.0);
    code = 1;
  }
  if (relay.failed()) {
    code = 1;
  }

  executor.cancel();
  spinner.join();
  // the chunk logger last, after the final commands
  if (chunks) {
    chunks->stop();
    if (chunks->samples_dropped() > 0 || chunks->write_failures() > 0) {
      log.post(
        Event::Warn, "the chunk logger dropped " + std::to_string(chunks->samples_dropped()) +
        " samples and failed to write " + std::to_string(chunks->write_failures()) +
        " chunks");
    }
  }
  log.post(Event::Info, "{\"event\": \"relay_stopped\"}");
  // A stalled stderr would hold rclcpp's console mutex for ever, and a relay thread that did
  // not return cannot be joined: leave at once, the commands and the log chunks being out.
  // _exit flushes no stdio stream, so it cannot block on one either.
  if (!log.drain(kLogDrainSeconds) || !joined) {
    _exit(code);
  }
  log.stop();
  heartbeat_timer.reset();
  node.reset();
  rclcpp::shutdown();
  return code;
}

}  // namespace
}  // namespace golem_safety

int main(int argc, char ** argv)
{
  // Every thread inherits this mask, so only main() takes the exit signals (sigtimedwait in
  // run()): Ctrl-C, ros2 launch shutting down (SIGINT, then SIGTERM), docker stop and a
  // closed terminal or SSH session (SIGHUP, whose default would end the process at once).
  sigset_t exit_signals;
  sigemptyset(&exit_signals);
  sigaddset(&exit_signals, SIGINT);
  sigaddset(&exit_signals, SIGTERM);
  sigaddset(&exit_signals, SIGHUP);
  pthread_sigmask(SIG_BLOCK, &exit_signals, nullptr);
  // A write to a closed stderr fails instead of ending the process mid-hand-back.
  signal(SIGPIPE, SIG_IGN);
  try {
    return golem_safety::run(argc, argv, exit_signals);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "safety_node failed: %s\n", e.what());
    return 1;
  }
}
