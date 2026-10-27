#include "golem_safety/safety_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

// Constructed at load time, so a latch never has to allocate it.
const std::string kReasonLost = "the estop latched, and its reason could not be recorded";

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

// A double as Python's json module writes a float (its repr): the fewest digits that read
// back as the same double, in fixed notation for decimal exponents -4..15 (with ".0" when
// the value is integral) and in scientific notation, with an exponent of at least two
// digits, otherwise; NaN, Infinity and -Infinity for the rest.
std::string python_float(double value)
{
  if (std::isnan(value)) {
    return "NaN";
  }
  if (std::isinf(value)) {
    return value > 0.0 ? "Infinity" : "-Infinity";
  }
  char text[40];
  for (int digits = 1; digits <= 17; ++digits) {
    std::snprintf(text, sizeof(text), "%.*e", digits - 1, value);
    if (std::strtod(text, nullptr) == value) {
      break;
    }
  }
  std::string scientific = text;  // [-]d[.ddd]e[+-]xx
  const bool negative = scientific[0] == '-';
  if (negative) {
    scientific.erase(0, 1);
  }
  const size_t e = scientific.find('e');
  const int exponent = std::atoi(scientific.c_str() + e + 1);
  std::string digits = scientific.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  while (digits.size() > 1 && digits.back() == '0') {
    digits.pop_back();
  }
  std::string out = negative ? "-" : "";
  if (exponent >= -4 && exponent < 16) {
    const size_t integral = static_cast<size_t>(exponent + 1);
    if (exponent < 0) {
      out += "0." + std::string(static_cast<size_t>(-exponent - 1), '0') + digits;
    } else if (digits.size() <= integral) {
      out += digits + std::string(integral - digits.size(), '0') + ".0";
    } else {
      out += digits.substr(0, integral) + "." + digits.substr(integral);
    }
  } else {
    out += digits.substr(0, 1);
    if (digits.size() > 1) {
      out += "." + digits.substr(1);
    }
    char exponent_text[16];
    std::snprintf(
      exponent_text, sizeof(exponent_text), "e%c%02d", exponent < 0 ? '-' : '+',
      std::abs(exponent));
    out += exponent_text;
  }
  return out;
}

// Python's round(value, 4), as the old layers rounded the logged weight and state age.
double round4(double value) {return std::round(value * 1e4) / 1e4;}

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
  auto use = [this](
    Input input, SlotRange slots, const std::string & topic, const std::string & source) {
      InputState & in = input_state(input);
      in.used = true;
      in.slots = slots;
      in.topic = topic;
      in.source = source;
      mode_inputs_.push_back(input);
    };
  switch (config_.mode) {
    case Mode::FullBody:
      use(Input::Full, SlotRange{0, last_body_slot}, config_.topics.low_cmd_in, "full_body_mode");
      break;
    case Mode::Split:
      use(
        Input::Lower, profile_.lower_slots, config_.topics.low_cmd_lower_in, "split_mode_lower");
      use(
        Input::Upper, profile_.upper_slots, config_.topics.low_cmd_upper_in, "split_mode_upper");
      break;
    case Mode::ArmSdk:
      use(Input::Upper, profile_.upper_slots, config_.topics.low_cmd_upper_in, "arm_sdk_mode");
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

const std::string & SafetyCore::estop_reason() const
{
  return reason_lost_ ? kReasonLost : estop_reason_;
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

std::string SafetyCore::joint(int slot) const
{
  return "motor " + std::to_string(slot) + " (" + profile_.joints[slot].name + ")";
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
        now, std::nullopt, [&] {
          return config_.topics.low_state + " mode_machine " +
          std::to_string(state.mode_machine) + " is not the " + profile_.robot + "'s " +
          std::to_string(profile_.mode_machine) + ": another robot or sim is on this domain";
        });
    }
    return;
  }
  if (config_.startup.rule == StartupRule::EngageGate && !engaged_) {
    return;
  }
  const auto violation = limit_violation(state);
  if (!violation) {
    return;
  }
  const Violation & v = *violation;
  latch(
    now, std::nullopt, [&] {
      if (!v.limit) {
        return joint(v.slot) + " " + v.field + " is " + number(v.value) + " in " +
        config_.topics.low_state;
      }
      return joint(v.slot) + " " + v.field + " out of estop range: " + number(v.value) +
      " (limit " + fixed(*v.limit, 3) + ", " + multiple(v.value, *v.limit) + ")";
    });
  raise(
    Event::Error, [&] {
      std::string diagnostic = "estop diagnostic, body slots in order:";
      auto add = [&](const char * name, auto value_of) {
          diagnostic += std::string("\n") + name + ": [";
          for (size_t slot = 0; slot < profile_.joints.size(); ++slot) {
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
      return diagnostic;
    });
}

// The first non-finite value on a checked slot, else the first q, then dq, then tau past
// its estop limit.
std::optional<SafetyCore::Violation> SafetyCore::limit_violation(const LowState & state) const
{
  const DerivedLimits & limits = config_.limits;
  for (const int slot : checked_slots_) {
    const auto & motor = state.motor_state[slot];
    const std::pair<const char *, float> fields[] = {
      {"q", motor.q}, {"dq", motor.dq}, {"tau", motor.tau_est}};
    for (const auto & [name, value] : fields) {
      if (!std::isfinite(value)) {
        return Violation{slot, name, value, std::nullopt};
      }
    }
  }
  for (const int slot : checked_slots_) {
    const double q = state.motor_state[slot].q;
    const double low = limits.q_estop_low[slot];
    const double high = limits.q_estop_high[slot];
    if (q < low || q > high) {
      return Violation{slot, "q", q, q < low ? low : high};
    }
  }
  for (const int slot : checked_slots_) {
    const double dq = state.motor_state[slot].dq;
    if (std::abs(dq) > limits.dq_estop[slot]) {
      return Violation{slot, "dq", dq, limits.dq_estop[slot]};
    }
  }
  for (const int slot : checked_slots_) {
    const double tau = state.motor_state[slot].tau_est;
    if (std::abs(tau) > limits.tau_estop[slot]) {
      return Violation{slot, "tau", tau, limits.tau_estop[slot]};
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
      raise(
        Event::Error, [&] {
          return std::string("ignoring commands on the ") + input_name(input) +
          " input, which " + mode_name(config_.mode) + " does not read";
        });
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
        now, std::nullopt, [&] {
          return "second writer on " + in.topic + ": a new writer sent a command " +
          fixed(quiet, 3) + " s after the current writer's last, and a handover needs a gap " +
          "over " + number(handover) + " s";
        });
      return;
    }
    in.writer = writer;
    raise(
      Event::Info, [&] {
        return "new writer on " + in.topic + " accepted: the previous writer's last command " +
        "was " + fixed(quiet, 3) + " s old (handover after " + number(handover) + " s)";
      });
  }
  in.writer_last = now;
  accept_command(in, cmd, now);
}

void SafetyCore::accept_command(InputState & in, const LowCmd & cmd, double now)
{
  if (const auto bad = non_finite_value(cmd)) {
    latch(
      now, std::nullopt, [&] {
        return "Command validation failed on " + in.topic + ": " + joint(bad->slot) + " " +
        bad->field + " is " + number(bad->value);
      });
    return;
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

// The first non-finite q, dq, tau, kp or kd on any body slot of `cmd`, in that field order.
std::optional<SafetyCore::BadValue> SafetyCore::non_finite_value(const LowCmd & cmd) const
{
  const std::pair<const char *, float MotorCmd::*> fields[] = {
    {"q", &MotorCmd::q}, {"dq", &MotorCmd::dq}, {"tau", &MotorCmd::tau}, {"kp", &MotorCmd::kp},
    {"kd", &MotorCmd::kd}};
  for (const auto & [name, member] : fields) {
    for (size_t slot = 0; slot < profile_.joints.size(); ++slot) {
      const float value = cmd.motor_cmd[slot].*member;
      if (!std::isfinite(value)) {
        return BadValue{static_cast<int>(slot), name, value};
      }
    }
  }
  return std::nullopt;
}

LowCmd SafetyCore::clipped(InputState & in, const LowCmd & cmd)
{
  const DerivedLimits & limits = config_.limits;
  LowCmd out = cmd;
  for (int slot = in.slots.first; slot <= in.slots.last; ++slot) {
    auto & motor = out.motor_cmd[slot];
    // The old layers' gain_clipped event: the commanded gain and its maximum.
    auto report_gain = [&](std::vector<bool> & reported, const char * gain, float value,
      double max) {
        if (!(value > max) || reported[slot]) {
          return;
        }
        reported[slot] = true;
        raise(
          Event::Warn, [&] {
            return "{\"commanded\": " + python_float(value) +
            ", \"event\": \"gain_clipped\", \"gain\": " + json_string(gain) +
            ", \"joint\": " + json_string(profile_.joints[slot].name) + ", \"max\": " +
            python_float(max) + ", \"source\": " + json_string(in.source) + "}";
          });
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
    latch(now, now, [] {return std::string("External estop triggered");});
  } else if (!plugged_in) {
    latch(now, now, [] {return std::string("External estop unplugged");});
  }
}

bool SafetyCore::set_latched(double now, std::optional<double> trigger) noexcept
{
  if (estopped()) {
    return false;
  }
  estopped_.store(true, std::memory_order_release);
  PendingReport report;
  report.decision = now;
  report.trigger = trigger;
  pending_report_ = report;
  if (config_.mode == Mode::ArmSdk && !release_.begun) {
    release_ = Release{};
    release_.begun = true;
    release_.from = weight_;
  }
  return true;
}

// The old layers' estop report, with sorted keys: estop_command_written, or
// estop_latched_silent when the first step after the latch wrote nothing.
std::string SafetyCore::estop_report(const PendingReport & report) const
{
  const long long decision_ms = ms(report.decision);
  const std::string trigger_ms =
    report.trigger ? std::to_string(ms(*report.trigger)) : std::string("null");
  const std::string reason = json_string(estop_reason());
  if (!report.written) {
    return "{\"decision_ms\": " + std::to_string(decision_ms) +
           ", \"event\": \"estop_latched_silent\", \"reason\": " + reason +
           ", \"trigger_stamp_ms\": " + trigger_ms + "}";
  }
  const long long action_ms = ms(report.action);
  std::string text = "{\"action_ms\": " + std::to_string(action_ms) + ", \"decision_ms\": " +
    std::to_string(decision_ms) + ", \"decision_to_action_ms\": " +
    std::to_string(action_ms - decision_ms) + ", \"event\": \"estop_command_written\", " +
    "\"reason\": " + reason + ", \"trigger_stamp_ms\": " + trigger_ms;
  if (report.trigger) {
    text += ", \"trigger_to_action_ms\": " + std::to_string(action_ms - ms(*report.trigger));
  }
  return text + "}";
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
// otherwise. A change between running and holding raises the old layers'
// arm_sdk_release_ramping or arm_sdk_release_held event, with the weight reached and the
// /lowstate age.
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
    raise(
      running ? Event::Info : Event::Warn, [&] {
        const std::string age =
        state_time_ ? python_float(round4(now - *state_time_)) : std::string("null");
        return std::string("{\"event\": ") +
        (running ? "\"arm_sdk_release_ramping\"" : "\"arm_sdk_release_held\"") +
        ", \"state_age\": " + age + ", \"weight\": " + python_float(round4(weight_)) + "}";
      });
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
  const size_t first_event = events_.size();
  std::optional<LowCmd> out;
  if (!took_control()) {
    out = std::nullopt;
  } else if (config_.mode == Mode::ArmSdk) {
    out = merged();
    set_weight(*out, release_weight(now, exiting));
  } else if (config_.estop_response.after == EstopAfter::BurstThenSilence) {
    if (burst_written_ < config_.estop_response.burst_count) {
      out = estop_cmd_;
      ++burst_written_;
    }
  } else {
    out = estop_cmd_;
  }
  if (out) {
    estop_written_ = true;
  }
  // The report is built by drain_events(), once this step's command is out.
  if (pending_report_ && !pending_report_->acted) {
    pending_report_->acted = true;
    pending_report_->written = out.has_value();
    pending_report_->action = now;
    pending_report_->event_index = first_event;
  }
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
    if (burst || !estop_written_) {
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
  if (pending_report_ && pending_report_->acted) {
    const PendingReport report = *pending_report_;
    pending_report_.reset();
    try {
      const size_t index = std::min(report.event_index, out.size());
      out.insert(
        out.begin() + static_cast<std::ptrdiff_t>(index),
        Event{Event::Error, estop_report(report)});
    } catch (...) {
    }
  }
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
