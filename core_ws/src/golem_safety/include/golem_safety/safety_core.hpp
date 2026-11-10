#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "golem_safety/chunk_logger.hpp"
#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

namespace golem_safety
{

// Name a command input: full_body_mode reads Full (every body slot); split_mode reads Lower
// (the profile's lower slots) and Upper (its upper slots).
enum class Input { Full, Upper, Lower };

// Identify the writer of a command by its DDS publisher GID, as raw bytes.
using WriterId = std::string;

// Carry one line for the node to log at `kind`; the core never logs by itself.
struct Event
{
  enum Kind { Info, Warn, Error } kind;
  std::string text;
};

// Make every decision of the safety layer, without ROS: the node feeds it the samples it
// receives and publishes what step() returns. `now` is s of robot time (RobotClock: the
// boot-time clock, or under use_sim_time a simulator's /clock) and must never go backwards.
//
// Relay. Each input keeps its newest valid command. step() starts from zero_command and
// copies onto each engaged input's slots that input's newest command, so a slot no engaged
// input owns stays at zero_command (every motor off). step() and exit_step() stamp the
// header and CRC on whatever they return, last.
//
// Engaging. Each input engages on its own first valid command that arrives while the newest
// /lowstate carries the profile's mode_machine and, with use_estop, a hardware estop sample
// has arrived (input_engaged, Info). A command that cannot engage raises engage_refused
// (Warn) with the first unmet condition, at most once a second. A /lowstate from another
// robot before engaging only delays engaging.
//
// Relay ready. relay_ready() is true while those input-independent conditions hold. Before
// engaging, a /lowstate from another robot makes it false again. A sample that latches never
// makes it true, and once latched it keeps its value: estopped() is what counts.
//
// SAFETY: a quiet input is held and reported, never latched. Once an input has engaged, a gap
// in its commands longer than its stale time (stale.upper_seconds for Upper,
// stale.lower_seconds for Lower and Full) raises input_quiet (Warn) while its last command
// keeps being relayed; its next command raises input_resumed (Info). Once any input has
// engaged, /lowstate is reported the same way after stale.state_seconds (state_quiet,
// state_resumed), and the relay keeps running. A stale time <= 0 turns its report off.
//
// SAFETY: a command is valid when q, dq, tau, kp and kd are finite on every body slot, the
// input's own or not; any other command latches the estop. A valid command is clipped on
// the input's slots before it is kept: q to [q_clip_low, q_clip_high], dq and tau to
// +-dq_clip and +-tau_clip, kp and kd to [0, kp_max] and [0, kd_max]. A kp or kd above its
// maximum raises gain_clipped (Warn) once per joint and gain.
//
// SAFETY: one writer per input. The writer of an input's first command is its current
// writer. A command from another writer becomes current (Info) only when the current
// writer's last command, valid or not, is older than single_writer.handover_seconds;
// otherwise it latches the estop with "second writer on <topic>".
//
// SAFETY: the /lowstate checks run on every sample that carries the profile's mode_machine,
// from the first one, before any input engages.
//
// SAFETY: the estop latches, one way until the process restarts, on
//   - a command with a non-finite value, before or after engaging: "Command validation
//     failed on <topic>: motor <slot> (<joint>) <field> is <value>";
//   - a /lowstate whose mode_machine is not the profile's, once an input has engaged;
//   - on ROS domain 0 with control.state_lag_ms > 0, a /lowstate that arrives more than
//     state_lag_ms late against the robot's uint32 ms tick: its arrival minus its tick,
//     less a baseline that is the smallest such offset seen and may rise by 0.5 ms/s to
//     follow clock drift;
//   - a /lowstate q outside [q_estop_low, q_estop_high], |dq| above dq_estop, |tau_est|
//     above tau_estop, or any of them non-finite, on any slot of the mode's inputs; the
//     reason names the joint, the value, the limit and the multiple of the limit;
//   - a second writer on an input;
//   - with use_estop, a hardware estop sample that is triggered or unplugged, and, from the
//     first sample on, a step() more than estop.timeout_seconds after the newest sample.
// The latch sets the flag before any other work, and nothing on its path can throw.
//
// SAFETY: after the latch, step() returns the estop command (estop_command()) and nothing
// else: estop_response.burst_count times and then nothing with burst_then_silence, on every
// step with keep_writing. The first step (or exit_step()) after the latch writes the estop
// command, and the drain_events() after it adds estop_command_written (Error): the reason,
// decision_ms (when the core latched), action_ms (that step's `now`) and their difference,
// plus, for the hardware estop, trigger_stamp_ms (the sample's arrival) and
// trigger_to_action_ms; all ms of robot time. It is built after the step so that
// nothing is formatted before the step's command is handed back.
//
// Events. engage_refused, input_engaged, input_quiet, input_resumed, state_quiet,
// state_resumed, gain_clipped and estop_command_written are one JSON object each: keys
// sorted, ", " and ": " separators, and numbers as Python's json module writes them, times
// rounded to 4 decimals. The other events are plain text. Raising an event never throws: an
// event that cannot be built is dropped.
//
// Threads: one thread calls every method, except estopped() and relay_ready(), which any
// thread may call.
class SafetyCore
{
public:
  // Build the core for one robot and config. `domain_id` is the ROS domain: the
  // transport-lag check runs only on 0, the real robots' command bus. Throws
  // std::invalid_argument when `config` is for another robot, or when estop_command() does.
  SafetyCore(RobotProfile profile, SafetyConfig config, bool use_estop, int domain_id);

  SafetyCore(const SafetyCore &) = delete;
  SafetyCore & operator=(const SafetyCore &) = delete;

  // Take one /lowstate sample: run its checks and update relay_ready().
  void on_state(const unitree_hg::msg::LowState & state, double now);
  // Take one command on `input` from `writer`. A command on an input the mode does not read
  // is ignored, with one Error event per input.
  void on_command(
    Input input, const unitree_hg::msg::LowCmd & cmd, const WriterId & writer, double now);
  // Take one hardware estop sample; ignored without use_estop.
  void on_estop(bool triggered, bool plugged_in, double now);

  // Return what to publish this cycle, header and CRC stamped; nullopt publishes nothing.
  std::optional<unitree_hg::msg::LowCmd> step(double now);

  // Return what to publish during shutdown, in place of step(); nullopt means done, and
  // step() answers as exit_step() from the first call on. Without an estop it is done at
  // once. Once estopped it first finishes the estop sequence: the rest of the burst with
  // burst_then_silence, and with keep_writing the estop command once if none has been
  // written since the latch.
  // SAFETY: after the last command the node stops publishing; the sims turn every motor off
  // after GOLEM_CMD_TIMEOUT of silence, unmeasured on the real H1-2 and G1.
  std::optional<unitree_hg::msg::LowCmd> exit_step(double now);

  // Return whether the estop has latched.
  bool estopped() const {return estopped_.load(std::memory_order_acquire);}
  // Return whether relaying can begin (see "Relay ready" above).
  bool relay_ready() const {return relay_ready_.load(std::memory_order_acquire);}
  // Return why the estop latched: empty before the latch, a fixed text when the reason
  // could not be built.
  const std::string & estop_reason() const;

  // Return the events raised since the last call, oldest first, with estop_command_written
  // in its place once the step after the latch has run.
  std::vector<Event> drain_events();

  // Return, once, the log sample of the newest /lowstate, or nullopt. With logging.enabled
  // there is one per /lowstate that arrived before the latch, the latching one included:
  // its motor fields and IMU, and the last command step() or exit_step() returned
  // (zero_command before any), on the profile's body slots. time_stamp is the wall clock at
  // the /lowstate's arrival.
  std::optional<LogSample> take_log_sample();

private:
  // One command input's configuration and state.
  struct InputState
  {
    bool used = false;  // the mode reads this input
    SlotRange slots;
    std::string topic;
    std::string source;          // the input's name in events, e.g. split_mode_upper
    double stale_seconds = 0.0;  // s; <= 0 turns the quiet report off
    std::optional<WriterId> writer;  // the current writer
    double writer_last = 0.0;        // s, arrival of the current writer's last command
    std::optional<unitree_hg::msg::LowCmd> newest;  // the newest valid command, clipped
    double newest_time = 0.0;                       // s, its arrival
    bool engaged = false;  // its slots carry `newest`
    bool quiet = false;    // input_quiet raised, input_resumed not yet
    bool misuse_reported = false;  // the "input the mode does not read" event was raised
  };

  // Name why a command cannot engage.
  struct Refusal
  {
    enum Kind { NoState, ModeMachine, NoEstopSample } kind;
    int mode_machine = 0;  // ModeMachine: the sample's
  };

  // Hold the transport-lag check's state: the robot's ms tick, unwrapped, against `now`.
  struct StateLag
  {
    std::optional<uint32_t> last_tick;
    long long wraps = 0;  // times the uint32 tick has wrapped
    std::optional<double> base_ms;  // the smallest arrival offset seen, drift allowed for
    double base_time = 0.0;  // s, when base_ms was last updated
  };

  // Hold the estop report's times, s; its reason is estop_reason().
  struct PendingReport
  {
    double decision = 0.0;
    std::optional<double> trigger;  // the hardware estop sample's arrival
    bool acted = false;      // the first step after the latch has run
    double action = 0.0;
    size_t event_index = 0;  // where the report goes among that step's events
  };

  // Describe a measured value past its estop limit (`limit` set), or a non-finite one.
  struct Violation
  {
    int slot = 0;
    const char * field = "";
    double value = 0.0;
    std::optional<double> limit;
  };

  // Describe a non-finite value in a command.
  struct BadValue
  {
    int slot = 0;
    const char * field = "";
    double value = 0.0;
  };

  // Return the state of `input`.
  InputState & input_state(Input input);

  // Run the /lowstate checks, which may latch.
  void check_state(const unitree_hg::msg::LowState & state, double now);
  // Store whether the input-independent engage conditions hold.
  void update_relay_ready();

  // Validate, clip and keep a command from the current writer, then try to engage `in`.
  void accept_command(InputState & in, const unitree_hg::msg::LowCmd & cmd, double now);
  // Engage `in` unless refusal() names a reason.
  void try_engage(InputState & in, double now);
  // Return the first unmet engage condition, or nullopt.
  std::optional<Refusal> refusal() const;
  // Return the engage_refused reason text of `refusal`.
  std::string refusal_text(const Refusal & refusal) const;
  // Raise engage_refused, at most once every kRefusalReportSeconds.
  void report_refusal(const Refusal & refusal, double now);
  // Raise input_quiet and state_quiet for gaps that have just passed their stale time.
  void report_quiet(double now);
  // Return how late, ms, a /lowstate with `tick` arrived at `now`.
  double state_lag_ms(uint32_t tick, double now);
  // Latch when the hardware estop has gone silent.
  void check_estop_input(double now);
  // Return the first non-finite value of `cmd` on a body slot, or nullopt.
  std::optional<BadValue> non_finite_value(const unitree_hg::msg::LowCmd & cmd) const;
  // Return `cmd` clipped to the limits on `in`'s slots, raising gain_clipped as needed.
  unitree_hg::msg::LowCmd clipped(InputState & in, const unitree_hg::msg::LowCmd & cmd);
  // Return the first measured value past its estop limit or not finite, or nullopt.
  std::optional<Violation> limit_violation(const unitree_hg::msg::LowState & state) const;
  // Return "motor <slot> (<joint name>)" for event and reason texts.
  std::string joint(int slot) const;

  // Latch with the reason build_reason() returns: the flag first, then the pending report
  // (set_latched), then the reason, whose failure leaves the fixed text. Does nothing once
  // latched.
  template<class BuildReason>
  void latch(double now, std::optional<double> trigger, BuildReason && build_reason) noexcept
  {
    if (!set_latched(now, trigger)) {
      return;
    }
    try {
      estop_reason_ = build_reason();
    } catch (...) {
      reason_lost_ = true;
    }
  }
  // Set the latch flag and the pending report; return false when already latched.
  bool set_latched(double now, std::optional<double> trigger) noexcept;
  // Return the estop_command_written JSON of `report`.
  std::string estop_report(const PendingReport & report) const;

  // Append the event build_text() returns; drop it when it cannot be built.
  template<class BuildText>
  void raise(Event::Kind kind, BuildText && build_text) noexcept
  {
    try {
      events_.push_back(Event{kind, build_text()});
    } catch (...) {
    }
  }

  // Return zero_command with each engaged input's newest command on its slots.
  unitree_hg::msg::LowCmd merged() const;
  // Return this step's estop command per estop_response, or nullopt once the burst is done.
  std::optional<unitree_hg::msg::LowCmd> estop_output(double now);
  // Stamp the header and CRC on `out` and remember it as the last command written.
  std::optional<unitree_hg::msg::LowCmd> finish(std::optional<unitree_hg::msg::LowCmd> out);

  const RobotProfile profile_;
  const SafetyConfig config_;
  const bool use_estop_;
  const unitree_hg::msg::LowCmd zero_cmd_;
  // SAFETY: built in the constructor, where a bad config throws, so that the latch path
  // only copies it and cannot fail.
  const unitree_hg::msg::LowCmd estop_cmd_;

  std::array<InputState, 3> inputs_;
  std::vector<Input> mode_inputs_;
  std::vector<int> checked_slots_;  // every slot of the mode's inputs, ascending: the
                                    // slots the /lowstate limit checks cover

  bool engaged_ = false;  // an input has engaged
  std::optional<double> refusal_reported_;  // s, the last engage_refused

  std::optional<unitree_hg::msg::LowState> state_;
  std::optional<double> state_time_;  // s, the last /lowstate's arrival, kept while estopped
  double state_wall_time_ = 0.0;  // s, wall clock at the newest /lowstate's arrival
  bool state_unlogged_ = false;   // take_log_sample() has a sample to return
  bool state_quiet_ = false;  // state_quiet raised, state_resumed not yet
  const bool lag_checked_;    // ROS domain 0 (the real robots' bus) and state_lag_ms > 0
  StateLag lag_;

  std::optional<double> estop_sample_time_;  // s, the newest hardware estop sample's arrival

  std::atomic<bool> relay_ready_{false};
  std::atomic<bool> estopped_{false};
  std::string estop_reason_;
  bool reason_lost_ = false;  // estop_reason() returns kReasonLost
  std::optional<PendingReport> pending_report_;
  int burst_written_ = 0;      // estop commands of the burst written, at most burst_count
  bool estop_written_ = false;  // an estop command has been written since the latch

  bool exiting_ = false;    // exit_step() has been called
  bool exit_done_ = false;  // exit_step() has returned nullopt

  std::vector<bool> kp_clip_reported_, kd_clip_reported_;  // gain_clipped raised, per slot
  unitree_hg::msg::LowCmd last_written_;  // what step() or exit_step() last returned
  std::vector<Event> events_;
};

}  // namespace golem_safety
