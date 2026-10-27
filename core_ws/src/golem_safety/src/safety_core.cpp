#include "golem_safety/safety_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "golem_safety/lowcmd.hpp"

namespace golem_safety
{

namespace
{

using unitree_hg::msg::LowCmd;
using unitree_hg::msg::LowState;
using unitree_hg::msg::MotorCmd;

// A dropped command is reported at most this often per input, s.
constexpr double kDropReportSeconds = 1.0;

size_t index_of(Input input) {return static_cast<size_t>(input);}

const char * input_name(Input input)
{
  switch (input) {
    case Input::Full:
      return "full";
    case Input::Upper:
      return "upper";
    case Input::Lower:
      return "lower";
  }
  return "unknown";
}

// %g-style, as short as the value allows: 2, 1.87, nan, -inf.
std::string number(double value)
{
  char text[32];
  std::snprintf(text, sizeof(text), "%.6g", value);
  return text;
}

std::string fixed(double value, int decimals)
{
  char text[48];
  std::snprintf(text, sizeof(text), "%.*f", decimals, value);
  return text;
}

// |value| as a multiple of a limit's magnitude, as "1.20x"; "inf x" for a zero limit.
std::string multiple(double value, double limit)
{
  if (limit == 0.0) {
    return "inf x";
  }
  return fixed(std::abs(value) / std::abs(limit), 2) + "x";
}

std::string json_string(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char escaped[8];
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
          out += escaped;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

// Milliseconds on the clock `now` is read from.
long long ms(double seconds) {return std::llround(seconds * 1000.0);}

// numpy.clip: the bounds are applied in this order, so a crossed pair yields `high`.
float clip(float value, double low, double high)
{
  return static_cast<float>(std::min(std::max(static_cast<double>(value), low), high));
}

// Fraction of a linear ramp of `seconds` done after `elapsed`, in [0, 1].
double ramp_fraction(double elapsed, double seconds)
{
  if (seconds <= 0.0) {
    return 1.0;
  }
  return std::min(std::max(elapsed / seconds, 0.0), 1.0);
}

// `config`, refused when it was read for another robot.
SafetyConfig for_robot(SafetyConfig config, const RobotProfile & profile)
{
  if (config.robot != profile.robot) {
    throw std::invalid_argument(
            "safety config " + config.source + " is for robot " + config.robot +
            ", not the " + profile.robot);
  }
  return config;
}

double wall_seconds()
{
  return std::chrono::duration<double>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

SafetyCore::SafetyCore(RobotProfile profile, SafetyConfig config, bool use_estop, int domain_id)
: profile_(std::move(profile)), config_(for_robot(std::move(config), profile_)),
  use_estop_(use_estop), domain_id_(domain_id), zero_cmd_(zero_command(profile_)),
  estop_cmd_(estop_command(profile_, config_, zero_cmd_)), last_written_(zero_cmd_)
{
  if (config_.mode == Mode::ArmSdk && !profile_.arm_sdk_weight_slot) {
    throw std::invalid_argument(
            "arm_sdk_mode needs a weight slot, and the " + profile_.robot +
            " profile has none");
  }

  const int last_body_slot = static_cast<int>(profile_.joints.size()) - 1;
  auto use = [this](Input input, SlotRange slots, const std::string & topic) {
      InputState & in = input_state(input);
      in.used = true;
      in.slots = slots;
      in.topic = topic;
      mode_inputs_.push_back(input);
    };
  switch (config_.mode) {
    case Mode::FullBody:
      use(Input::Full, SlotRange{0, last_body_slot}, config_.topics.low_cmd_in);
      break;
    case Mode::Split:
      use(Input::Lower, profile_.lower_slots, config_.topics.low_cmd_lower_in);
      use(Input::Upper, profile_.upper_slots, config_.topics.low_cmd_upper_in);
      break;
    case Mode::ArmSdk:
      use(Input::Upper, profile_.upper_slots, config_.topics.low_cmd_upper_in);
      break;
  }
  for (int slot = 0; slot <= last_body_slot; ++slot) {
    for (const Input input : mode_inputs_) {
      if (input_state(input).slots.contains(slot)) {
        checked_slots_.push_back(slot);
        break;
      }
    }
  }
  kp_clip_reported_.assign(profile_.joints.size(), false);
  kd_clip_reported_.assign(profile_.joints.size(), false);
}

SafetyCore::InputState & SafetyCore::input_state(Input input)
{
  return inputs_[index_of(input)];
}

// Whether the layer has been writing commands: from the first step with zero_stream, from
// engagement with engage_gate.
bool SafetyCore::took_control() const
{
  return config_.startup.rule == StartupRule::ZeroStream || engaged_;
}

// Whether the robot's /lowstate reaches the layer, so that what the layer writes reaches
// the robot too.
bool SafetyCore::state_flowing(double now) const
{
  if (!state_time_) {
    return false;
  }
  return config_.stale.state_seconds <= 0.0 || now - *state_time_ <= config_.stale.state_seconds;
}

void SafetyCore::on_state(const LowState & state, double now)
{
  state_time_ = now;
  if (estopped()) {
    return;
  }
  state_ = state;
  state_wall_time_ = wall_seconds();
  state_unlogged_ = config_.logging.enabled;

  if (state.mode_machine != profile_.mode_machine) {
    if (engaged_) {
      latch(
        config_.topics.low_state + " mode_machine " + std::to_string(state.mode_machine) +
        " is not the " + profile_.robot + "'s " + std::to_string(profile_.mode_machine) +
        ": another robot or sim is on this domain", now, std::nullopt);
    }
    return;
  }
  if (config_.startup.rule == StartupRule::EngageGate && !engaged_) {
    return;
  }
  if (const auto reason = limit_violation(state)) {
    latch(*reason, now, std::nullopt);
    const size_t joints = profile_.joints.size();
    std::string diagnostic = "estop diagnostic, body slots in order:";
    auto add = [&](const char * name, auto value_of) {
        diagnostic += std::string("\n") + name + ": [";
        for (size_t slot = 0; slot < joints; ++slot) {
          diagnostic += (slot ? ", " : "") + number(value_of(slot));
        }
        diagnostic += "]";
      };
    add("q_cmd", [&](size_t i) {return last_written_.motor_cmd[i].q;});
    add("dq_cmd", [&](size_t i) {return last_written_.motor_cmd[i].dq;});
    add("tau_cmd", [&](size_t i) {return last_written_.motor_cmd[i].tau;});
    add("q", [&](size_t i) {return state.motor_state[i].q;});
    add("dq", [&](size_t i) {return state.motor_state[i].dq;});
    add("ddq", [&](size_t i) {return state.motor_state[i].ddq;});
    events_.push_back({Event::Error, diagnostic});
  }
}

std::optional<std::string> SafetyCore::limit_violation(const LowState & state) const
{
  const DerivedLimits & limits = config_.limits;
  auto joint = [this](int slot) {
      return "motor " + std::to_string(slot) + " (" + profile_.joints[slot].name + ")";
    };
  for (const int slot : checked_slots_) {
    const auto & motor = state.motor_state[slot];
    const std::pair<const char *, float> fields[] = {
      {"q", motor.q}, {"dq", motor.dq}, {"tau", motor.tau_est}};
    for (const auto & [name, value] : fields) {
      if (!std::isfinite(value)) {
        return joint(slot) + " " + name + " is " + number(value) + " in " +
               config_.topics.low_state;
      }
    }
  }
  for (const int slot : checked_slots_) {
    const double q = state.motor_state[slot].q;
    const double low = limits.q_estop_low[slot];
    const double high = limits.q_estop_high[slot];
    if (q < low || q > high) {
      const double limit = q < low ? low : high;
      return joint(slot) + " q out of estop range: " + number(q) + " (limit " +
             fixed(limit, 3) + ", " + multiple(q, limit) + ")";
    }
  }
  for (const int slot : checked_slots_) {
    const double dq = state.motor_state[slot].dq;
    const double limit = limits.dq_estop[slot];
    if (std::abs(dq) > limit) {
      return joint(slot) + " dq out of estop range: " + number(dq) + " (limit " +
             fixed(limit, 3) + ", " + multiple(dq, limit) + ")";
    }
  }
  for (const int slot : checked_slots_) {
    const double tau = state.motor_state[slot].tau_est;
    const double limit = limits.tau_estop[slot];
    if (std::abs(tau) > limit) {
      return joint(slot) + " tau out of estop range: " + number(tau) + " (limit " +
             fixed(limit, 3) + ", " + multiple(tau, limit) + ")";
    }
  }
  return std::nullopt;
}

void SafetyCore::on_command(Input input, const LowCmd & cmd, const WriterId & writer, double now)
{
  if (estopped() || exiting_) {
    return;
  }
  InputState & in = input_state(input);
  if (!in.used) {
    if (!in.misuse_reported) {
      in.misuse_reported = true;
      events_.push_back(
        {Event::Error, std::string("ignoring commands on the ") + input_name(input) +
          " input, which " + mode_name(config_.mode) + " does not read"});
    }
    return;
  }

  if (!in.writer) {
    in.writer = writer;
  } else if (*in.writer != writer) {
    const double quiet = now - in.writer_last;
    const double handover = config_.single_writer.handover_seconds;
    if (!(quiet > handover)) {
      latch(
        "second writer on " + in.topic + ": a new writer sent a command " + fixed(quiet, 3) +
        " s after the current writer's last, and a handover needs a gap over " +
        number(handover) + " s", now, std::nullopt);
      return;
    }
    in.writer = writer;
    events_.push_back(
      {Event::Info, "new writer on " + in.topic + " accepted: the previous writer's last "
        "command was " + fixed(quiet, 3) + " s old (handover after " + number(handover) +
        " s)"});
  }
  in.writer_last = now;
  accept_command(in, cmd, now);
}

void SafetyCore::accept_command(InputState & in, const LowCmd & cmd, double now)
{
  const std::pair<const char *, float MotorCmd::*> fields[] = {
    {"q", &MotorCmd::q}, {"dq", &MotorCmd::dq}, {"tau", &MotorCmd::tau}, {"kp", &MotorCmd::kp},
    {"kd", &MotorCmd::kd}};
  for (const auto & [name, member] : fields) {
    for (int slot = in.slots.first; slot <= in.slots.last; ++slot) {
      const float value = cmd.motor_cmd[slot].*member;
      if (!std::isfinite(value)) {
        report_drop(
          in, "motor " + std::to_string(slot) + " (" + profile_.joints[slot].name + ") " +
          name + " is " + number(value), now);
        return;
      }
    }
  }

  in.newest = clipped(in, cmd);
  in.newest_time = now;
  if (engaged_) {
    return;
  }
  if (config_.startup.rule == StartupRule::ZeroStream) {
    engaged_ = true;
  } else {
    engaged_ = std::all_of(
      mode_inputs_.begin(), mode_inputs_.end(),
      [this](Input input) {return input_state(input).newest.has_value();});
  }
  if (engaged_ && config_.mode == Mode::ArmSdk) {
    engage_start_ = now;
  }
}

void SafetyCore::report_drop(InputState & in, const std::string & detail, double now)
{
  ++in.drops_unreported;
  if (in.last_drop_report && now - *in.last_drop_report < kDropReportSeconds) {
    return;
  }
  const size_t count = in.drops_unreported;
  std::string text = "dropped " +
    (count == 1 ? std::string("a command") : std::to_string(count) + " commands") + " on " +
    in.topic + " with a non-finite value (latest: " + detail + "); ";
  text += in.newest ? "its last valid command is still relayed" :
    "the input has no valid command yet";
  events_.push_back({Event::Warn, text});
  in.last_drop_report = now;
  in.drops_unreported = 0;
}

LowCmd SafetyCore::clipped(InputState & in, const LowCmd & cmd)
{
  const DerivedLimits & limits = config_.limits;
  LowCmd out = cmd;
  for (int slot = in.slots.first; slot <= in.slots.last; ++slot) {
    auto & motor = out.motor_cmd[slot];
    auto report_gain = [&](std::vector<bool> & reported, const char * gain, float value,
      double max) {
        if (value > max && !reported[slot]) {
          reported[slot] = true;
          events_.push_back(
            {Event::Warn, in.topic + " commands " + gain + " " + number(value) + " on motor " +
              std::to_string(slot) + " (" + profile_.joints[slot].name + "), above " + gain +
              "_max " + number(max) + ": clipped (reported once per joint and gain)"});
        }
      };
    report_gain(kp_clip_reported_, "kp", motor.kp, limits.kp_max);
    report_gain(kd_clip_reported_, "kd", motor.kd, limits.kd_max);
    motor.q = clip(motor.q, limits.q_clip_low[slot], limits.q_clip_high[slot]);
    motor.dq = clip(motor.dq, -limits.dq_clip[slot], limits.dq_clip[slot]);
    motor.tau = clip(motor.tau, -limits.tau_clip[slot], limits.tau_clip[slot]);
    motor.kp = clip(motor.kp, 0.0, limits.kp_max);
    motor.kd = clip(motor.kd, 0.0, limits.kd_max);
  }
  return out;
}

void SafetyCore::on_estop(bool triggered, bool plugged_in, double now)
{
  if (!use_estop_ || estopped()) {
    return;
  }
  if (triggered) {
    latch("External estop triggered", now, now);
  } else if (!plugged_in) {
    latch("External estop unplugged", now, now);
  }
}

void SafetyCore::latch(const std::string & reason, double now, std::optional<double> trigger)
{
  if (estopped()) {
    return;
  }
  estop_reason_ = reason;
  pending_report_ = PendingReport{reason, now, trigger};
  if (config_.mode == Mode::ArmSdk && !release_.begun) {
    release_ = Release{};
    release_.begun = true;
    release_.from = weight_;
  }
  estopped_.store(true, std::memory_order_release);
}

// The one-time estop report, raised with the first output after the latch: written, or
// nothing to write because the layer never took control.
void SafetyCore::report_estop(bool written, double now)
{
  if (!pending_report_) {
    return;
  }
  const PendingReport & report = *pending_report_;
  const long long decision_ms = ms(report.decision);
  const std::string trigger_ms =
    report.trigger ? std::to_string(ms(*report.trigger)) : std::string("null");
  std::string text = "{";
  if (written) {
    const long long action_ms = ms(now);
    text += "\"action_ms\": " + std::to_string(action_ms) + ", \"decision_ms\": " +
      std::to_string(decision_ms) + ", \"decision_to_action_ms\": " +
      std::to_string(action_ms - decision_ms) + ", \"event\": \"estop_command_written\", " +
      "\"reason\": " + json_string(report.reason) + ", \"trigger_stamp_ms\": " + trigger_ms;
    if (report.trigger) {
      text += ", \"trigger_to_action_ms\": " + std::to_string(action_ms - ms(*report.trigger));
    }
  } else {
    text += "\"decision_ms\": " + std::to_string(decision_ms) +
      ", \"event\": \"estop_latched_silent\", \"reason\": " + json_string(report.reason) +
      ", \"trigger_stamp_ms\": " + trigger_ms;
  }
  text += "}";
  events_.push_back({Event::Error, text});
  pending_report_.reset();
}

LowCmd SafetyCore::merged() const
{
  LowCmd out = zero_cmd_;
  for (const Input input : mode_inputs_) {
    const InputState & in = inputs_[index_of(input)];
    if (!in.newest) {
      continue;
    }
    for (int slot = in.slots.first; slot <= in.slots.last; ++slot) {
      out.motor_cmd[slot] = in.newest->motor_cmd[slot];
    }
  }
  return out;
}

void SafetyCore::set_weight(LowCmd & cmd, double weight) const
{
  auto & motor = cmd.motor_cmd[*profile_.arm_sdk_weight_slot];
  motor.mode = 0;
  motor.q = static_cast<float>(weight);
  motor.dq = 0.0F;
  motor.tau = 0.0F;
  motor.kp = 0.0F;
  motor.kd = 0.0F;
}

// The arm_sdk weight while handing the waist and arms back: from release_.from to 0 over
// release_seconds, advancing only while /lowstate flows or while exiting, holding
// otherwise. A change between running and holding raises an event.
double SafetyCore::release_weight(double now, bool exiting)
{
  const bool flowing = exiting || state_flowing(now);
  if (flowing && !release_.start) {
    release_.start = now - release_.elapsed;
  } else if (!flowing && release_.start) {
    release_.elapsed = now - *release_.start;
    release_.start.reset();
  }
  const bool running = release_.start.has_value();
  if (!release_.running || *release_.running != running) {
    release_.running = running;
    if (running) {
      events_.push_back(
        {Event::Info, "arm_sdk release ramp running at weight " + fixed(weight_, 4) +
          ": the weight falls to 0 over " + number(config_.arm_sdk.release_seconds) + " s"});
    } else {
      const std::string age = state_time_ ?
        fixed(now - *state_time_, 3) + " s old" : std::string("never received");
      events_.push_back(
        {Event::Warn, "arm_sdk release ramp held at weight " + fixed(weight_, 4) + ": " +
          config_.topics.low_state + " is " + age});
    }
  }
  if (running) {
    weight_ = release_.from *
      (1.0 - ramp_fraction(now - *release_.start, config_.arm_sdk.release_seconds));
  }
  return weight_;
}

std::optional<LowCmd> SafetyCore::relay_output(double now)
{
  if (!took_control()) {
    return std::nullopt;
  }
  LowCmd out = merged();
  if (config_.mode == Mode::ArmSdk) {
    weight_ = engage_start_ ?
      ramp_fraction(now - *engage_start_, config_.arm_sdk.engage_seconds) : 0.0;
    set_weight(out, weight_);
  }
  return out;
}

std::optional<LowCmd> SafetyCore::estop_output(double now, bool exiting)
{
  std::optional<LowCmd> out;
  if (!took_control()) {
    out = std::nullopt;
  } else if (config_.mode == Mode::ArmSdk) {
    out = merged();
    set_weight(*out, release_weight(now, exiting));
  } else if (config_.estop_response.after == EstopAfter::BurstThenSilence) {
    if (estop_commands_written_ < config_.estop_response.burst_count) {
      out = estop_cmd_;
    }
  } else {
    out = estop_cmd_;
  }
  if (out) {
    ++estop_commands_written_;
  }
  report_estop(out.has_value(), now);
  return out;
}

std::optional<LowCmd> SafetyCore::finish(std::optional<LowCmd> out)
{
  if (out) {
    stamp_header_and_crc(*out, profile_);
    last_written_ = *out;
  }
  return out;
}

std::optional<LowCmd> SafetyCore::step(double now)
{
  if (exiting_) {
    return exit_step(now);
  }
  return finish(estopped() ? estop_output(now, false) : relay_output(now));
}

std::optional<LowCmd> SafetyCore::exit_step(double now)
{
  exiting_ = true;
  if (exit_done_) {
    return std::nullopt;
  }
  std::optional<LowCmd> out;
  if (config_.mode == Mode::ArmSdk) {
    if (took_control() && engage_start_) {
      if (!release_.begun) {
        release_ = Release{};
        release_.begun = true;
        release_.from = weight_;
      }
      if (estopped()) {
        out = estop_output(now, true);
      } else {
        out = merged();
        set_weight(*out, release_weight(now, true));
      }
      exit_done_ = weight_ == 0.0;
    }
  } else if (estopped()) {
    const bool burst = config_.estop_response.after == EstopAfter::BurstThenSilence;
    if (burst || estop_commands_written_ == 0) {
      out = estop_output(now, true);
    }
    exit_done_ = !burst;
  }
  if (!out) {
    exit_done_ = true;
  }
  return finish(out);
}

std::vector<Event> SafetyCore::drain_events()
{
  std::vector<Event> out;
  out.swap(events_);
  return out;
}

std::optional<LogSample> SafetyCore::take_log_sample()
{
  if (!state_unlogged_ || !state_) {
    return std::nullopt;
  }
  state_unlogged_ = false;
  const LowState & state = *state_;
  const size_t joints = profile_.joints.size();
  LogSample sample(joints);
  sample.time_stamp = state_wall_time_;
  for (size_t slot = 0; slot < joints; ++slot) {
    const auto & motor = state.motor_state[slot];
    sample.mode[slot] = motor.mode;
    sample.q[slot] = motor.q;
    sample.dq[slot] = motor.dq;
    sample.ddq[slot] = motor.ddq;
    sample.tau_est[slot] = motor.tau_est;
    sample.temperature[slot] = {motor.temperature[0], motor.temperature[1]};
    sample.vol[slot] = motor.vol;
    sample.sensor[slot] = {motor.sensor[0], motor.sensor[1]};
    sample.motorstate[slot] = motor.motorstate;
    sample.reserve[slot] = {motor.reserve[0], motor.reserve[1], motor.reserve[2],
      motor.reserve[3]};
    const auto & cmd = last_written_.motor_cmd[slot];
    sample.q_cmd[slot] = cmd.q;
    sample.dq_cmd[slot] = cmd.dq;
    sample.tau_cmd[slot] = cmd.tau;
    sample.kp_cmd[slot] = cmd.kp;
    sample.kd_cmd[slot] = cmd.kd;
  }
  const auto & imu = state.imu_state;
  std::copy(imu.quaternion.begin(), imu.quaternion.end(), sample.imu_quaternion.begin());
  std::copy(imu.gyroscope.begin(), imu.gyroscope.end(), sample.imu_gyroscope.begin());
  std::copy(imu.accelerometer.begin(), imu.accelerometer.end(), sample.imu_accelerometer.begin());
  std::copy(imu.rpy.begin(), imu.rpy.end(), sample.imu_rpy.begin());
  sample.imu_temperature = imu.temperature;
  return sample;
}

}  // namespace golem_safety
