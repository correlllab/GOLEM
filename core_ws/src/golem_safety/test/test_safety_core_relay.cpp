#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "golem_safety/config.hpp"
#include "golem_safety/lowcmd.hpp"
#include "golem_safety/robot_profile.hpp"
#include "golem_safety/safety_core.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

using golem_safety::Event;
using golem_safety::Input;
using golem_safety::RobotProfile;
using golem_safety::SafetyConfig;
using golem_safety::SafetyCore;
using unitree_hg::msg::LowCmd;
using unitree_hg::msg::LowState;

// Every test feeds a /lowstate carrying the profile's mode_machine before any command,
// and on the G1 sends commands within engage.max_position_error of the measured q, so the
// tests hold under the start-up rules too.

namespace {

constexpr int kDomain = 91;
constexpr double kDt = 0.002;  // s, one 500 Hz cycle

const RobotProfile & h12()
{
  static const RobotProfile profile = golem_safety::load_robot_profile("h12");
  return profile;
}

const RobotProfile & g1()
{
  static const RobotProfile profile = golem_safety::load_robot_profile("g1");
  return profile;
}

SafetyConfig config(const RobotProfile & profile, const std::string & name)
{
  return golem_safety::load_safety_config(name, profile);
}

// A /lowstate of `profile`'s robot with every joint at rest at q.
LowState state_of(const RobotProfile & profile, float q = 0.0F)
{
  LowState state;
  state.mode_pr = profile.mode_pr;
  state.mode_machine = profile.mode_machine;
  for (auto & motor : state.motor_state) {
    motor.mode = 1;
    motor.q = q;
    motor.dq = 0.0F;
    motor.tau_est = 0.0F;
  }
  return state;
}

// A controller's command: every body slot enabled at q with kp and kd, the rest as
// zero_command has them.
LowCmd command(const RobotProfile & profile, float q = 0.0F, float kp = 60.0F, float kd = 2.0F)
{
  LowCmd cmd = golem_safety::zero_command(profile);
  for (size_t slot = 0; slot < profile.joints.size(); ++slot) {
    cmd.motor_cmd[slot].mode = 1;
    cmd.motor_cmd[slot].q = q;
    cmd.motor_cmd[slot].kp = kp;
    cmd.motor_cmd[slot].kd = kd;
  }
  return cmd;
}

// Sets every field of slots first..last to nonzero values, and the message's reserve words.
void fill_junk(LowCmd & cmd, int first, int last)
{
  for (int slot = first; slot <= last; ++slot) {
    auto & motor = cmd.motor_cmd[slot];
    motor.mode = 1;
    motor.q = 0.3F;
    motor.dq = 0.1F;
    motor.tau = 0.2F;
    motor.kp = 20.0F;
    motor.kd = 1.0F;
    motor.reserve = 7;
  }
  cmd.reserve = {1, 2, 3, 4};
}

size_t count(const std::vector<Event> & events, Event::Kind kind)
{
  size_t n = 0;
  for (const auto & event : events) {
    n += event.kind == kind ? 1 : 0;
  }
  return n;
}

bool contains(const std::string & text, const std::string & part)
{
  return text.find(part) != std::string::npos;
}

bool has_event(const std::vector<Event> & events, Event::Kind kind, const std::string & part)
{
  for (const auto & event : events) {
    if (event.kind == kind && contains(event.text, part)) {
      return true;
    }
  }
  return false;
}

float & field_of(unitree_hg::msg::MotorCmd & motor, const std::string & name)
{
  if (name == "q") {
    return motor.q;
  }
  if (name == "dq") {
    return motor.dq;
  }
  if (name == "tau") {
    return motor.tau;
  }
  if (name == "kp") {
    return motor.kp;
  }
  return motor.kd;
}

std::string texts(const std::vector<Event> & events)
{
  std::string out;
  for (const auto & event : events) {
    out += "[" + std::to_string(static_cast<int>(event.kind)) + "] " + event.text + "\n";
  }
  return out;
}

// Each JSON event, parsed, with its "event" equal to `name`: (the event, its object).
std::vector<std::pair<Event, YAML::Node>> named(
  const std::vector<Event> & events, const std::string & name)
{
  std::vector<std::pair<Event, YAML::Node>> out;
  for (const auto & event : events) {
    if (event.text.empty() || event.text[0] != '{') {
      continue;
    }
    YAML::Node node;
    try {
      node = YAML::Load(event.text);
    } catch (const YAML::Exception & e) {
      ADD_FAILURE() << "not JSON: " << event.text << " (" << e.what() << ")";
      continue;
    }
    if (node.IsMap() && node["event"] && node["event"].as<std::string>() == name) {
      out.emplace_back(event, node);
    }
  }
  return out;
}

// A JSON object's keys, in the order written.
std::vector<std::string> keys_of(const YAML::Node & node)
{
  std::vector<std::string> keys;
  for (const auto & entry : node) {
    keys.push_back(entry.first.as<std::string>());
  }
  return keys;
}

// Every field equal, bit for bit.
void expect_same(const LowCmd & actual, const LowCmd & expected)
{
  EXPECT_EQ(actual.mode_pr, expected.mode_pr);
  EXPECT_EQ(actual.mode_machine, expected.mode_machine);
  for (size_t slot = 0; slot < expected.motor_cmd.size(); ++slot) {
    SCOPED_TRACE("slot " + std::to_string(slot));
    EXPECT_EQ(actual.motor_cmd[slot], expected.motor_cmd[slot]);
  }
  EXPECT_EQ(actual.reserve, expected.reserve);
  EXPECT_EQ(actual.crc, expected.crc);
}

void expect_stamped(const LowCmd & cmd, const RobotProfile & profile)
{
  EXPECT_EQ(cmd.mode_machine, profile.mode_machine);
  EXPECT_EQ(cmd.mode_pr, profile.mode_pr);
  EXPECT_EQ(cmd.crc, golem_safety::lowcmd_crc(cmd));
}

// The estop on a measured q past the estop range of `slot`.
LowState past_estop(const RobotProfile & profile, const SafetyConfig & cfg, int slot)
{
  LowState state = state_of(profile);
  state.motor_state[slot].q = static_cast<float>(cfg.limits.q_estop_high[slot] + 0.2);
  return state;
}

// G1 arm_sdk, engaged at t = 0 and fed /lowstate every 10 ms until `until`; returns the
// last command written.
LowCmd engage_arm_sdk(SafetyCore & core, const LowCmd & upper, double until)
{
  std::optional<LowCmd> out;
  for (int i = 0; i * 0.01 <= until + 1e-9; ++i) {
    const double t = i * 0.01;
    core.on_state(state_of(g1()), t);
    core.on_command(Input::Upper, upper, "upper", t);
    out = core.step(t);
  }
  return *out;
}

// H1-2 split, both inputs engaged at t = 0: Lower from writer "lower", Upper from "A".
struct H12Split
{
  SafetyConfig cfg = config(h12(), "sim_safety_split");
  SafetyCore core{h12(), cfg, false, kDomain};

  H12Split()
  {
    core.on_state(state_of(h12()), 0.0);
    core.on_command(Input::Lower, command(h12()), "lower", 0.0);
    core.on_command(Input::Upper, command(h12()), "A", 0.0);
  }
};

}  // namespace

TEST(Relay, SplitMergesUpperAndLowerSlots)
{
  SafetyCore core(h12(), config(h12(), "sim_safety_split"), false, kDomain);
  core.on_state(state_of(h12()), 0.0);
  LowCmd lower = command(h12());
  lower.motor_cmd[5].q = 0.1F;
  lower.motor_cmd[20].q = 0.9F;
  lower.mode_machine = 0;
  LowCmd upper = command(h12());
  upper.motor_cmd[5].q = -0.2F;
  upper.motor_cmd[20].q = 0.5F;
  upper.mode_machine = 9;
  core.on_command(Input::Lower, lower, "lower", 0.001);
  core.on_command(Input::Upper, upper, "upper", 0.001);

  const auto out = core.step(0.002);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd[5].q, 0.1F);
  EXPECT_EQ(out->motor_cmd[20].q, 0.5F);
  for (int slot = 0; slot <= 11; ++slot) {
    EXPECT_EQ(out->motor_cmd[slot], lower.motor_cmd[slot]) << "slot " << slot;
  }
  for (int slot = 12; slot <= 26; ++slot) {
    EXPECT_EQ(out->motor_cmd[slot], upper.motor_cmd[slot]) << "slot " << slot;
  }
  expect_stamped(*out, h12());
  EXPECT_EQ(out->mode_machine, 6);

  // The G1 likewise, on its own slots and header.
  SafetyCore g1_core(g1(), config(g1(), "sim_split"), false, kDomain);
  g1_core.on_state(state_of(g1()), 0.0);
  LowCmd g1_lower = command(g1());
  g1_lower.motor_cmd[5].q = 0.1F;
  LowCmd g1_upper = command(g1());
  g1_upper.motor_cmd[20].q = 0.05F;
  g1_core.on_command(Input::Lower, g1_lower, "lower", 0.001);
  g1_core.on_command(Input::Upper, g1_upper, "upper", 0.001);
  const auto g1_out = g1_core.step(0.002);
  ASSERT_TRUE(g1_out);
  EXPECT_EQ(g1_out->motor_cmd[5].q, 0.1F);
  EXPECT_EQ(g1_out->motor_cmd[20].q, 0.05F);
  for (int slot = 12; slot <= 28; ++slot) {
    EXPECT_EQ(g1_out->motor_cmd[slot], g1_upper.motor_cmd[slot]) << "slot " << slot;
  }
  expect_stamped(*g1_out, g1());
  EXPECT_EQ(g1_out->mode_machine, 5);
}

TEST(Relay, UnownedSlotsAreZeroCommand)
{
  const LowCmd zero = golem_safety::zero_command(h12());

  // split_mode: Lower owns 0-11 and Upper 12-26; 27-34 are no input's.
  SafetyCore split(h12(), config(h12(), "sim_safety_split"), false, kDomain);
  split.on_state(state_of(h12()), 0.0);
  LowCmd lower = command(h12());
  fill_junk(lower, 27, 34);
  LowCmd upper = command(h12());
  fill_junk(upper, 27, 34);
  split.on_command(Input::Lower, lower, "lower", 0.001);
  split.on_command(Input::Upper, upper, "upper", 0.001);
  const auto split_out = split.step(0.002);
  ASSERT_TRUE(split_out);
  for (int slot = 27; slot <= 34; ++slot) {
    EXPECT_EQ(split_out->motor_cmd[slot], zero.motor_cmd[slot]) << "slot " << slot;
  }
  EXPECT_EQ(split_out->reserve, zero.reserve);

  // full_body_mode: Full owns 0-26.
  SafetyCore full(h12(), config(h12(), "default_safety_full"), false, kDomain);
  full.on_state(state_of(h12()), 0.0);
  LowCmd full_cmd = command(h12());
  fill_junk(full_cmd, 27, 34);
  full.on_command(Input::Full, full_cmd, "full", 0.001);
  const auto full_out = full.step(0.002);
  ASSERT_TRUE(full_out);
  for (int slot = 0; slot <= 26; ++slot) {
    EXPECT_EQ(full_out->motor_cmd[slot], full_cmd.motor_cmd[slot]) << "slot " << slot;
  }
  for (int slot = 27; slot <= 34; ++slot) {
    EXPECT_EQ(full_out->motor_cmd[slot], zero.motor_cmd[slot]) << "slot " << slot;
  }
  EXPECT_EQ(full_out->reserve, zero.reserve);

  // arm_sdk_mode: Upper owns 12-28; the legs and 30-34 are zero_command's, and slot 29
  // carries only the weight (0.5 halfway through the 1 s engage ramp).
  const LowCmd g1_zero = golem_safety::zero_command(g1());
  SafetyCore arm_sdk(g1(), config(g1(), "sim_arm_sdk"), false, kDomain);
  arm_sdk.on_state(state_of(g1()), 0.0);
  LowCmd arm = command(g1());
  fill_junk(arm, 0, 11);
  fill_junk(arm, 29, 34);
  arm_sdk.on_command(Input::Upper, arm, "upper", 0.0);
  arm_sdk.on_state(state_of(g1()), 0.5);
  const auto arm_out = arm_sdk.step(0.5);
  ASSERT_TRUE(arm_out);
  for (int slot = 0; slot <= 11; ++slot) {
    EXPECT_EQ(arm_out->motor_cmd[slot], g1_zero.motor_cmd[slot]) << "slot " << slot;
  }
  for (int slot = 30; slot <= 34; ++slot) {
    EXPECT_EQ(arm_out->motor_cmd[slot], g1_zero.motor_cmd[slot]) << "slot " << slot;
  }
  auto weight_slot = arm_out->motor_cmd[29];
  EXPECT_NEAR(weight_slot.q, 0.5, 1e-6);
  weight_slot.q = 0.0F;
  EXPECT_EQ(weight_slot, g1_zero.motor_cmd[29]);
  EXPECT_EQ(arm_out->reserve, g1_zero.reserve);
  expect_stamped(*arm_out, g1());
}

TEST(Clip, PositionVelocityTorqueAndGains)
{
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  const auto & limits = cfg.limits;
  SafetyCore core(h12(), cfg, false, kDomain);
  core.on_state(state_of(h12()), 0.0);

  LowCmd upper = command(h12());
  auto & high = upper.motor_cmd[20];  // every field above its limit
  high.q = 10.0F;
  high.dq = 1000.0F;
  high.tau = -1000.0F;
  high.kp = 1000.0F;
  high.kd = 100.0F;
  auto & low = upper.motor_cmd[21];  // every field below its limit
  low.q = -10.0F;
  low.dq = -1000.0F;
  low.tau = 1000.0F;
  low.kp = -5.0F;
  low.kd = -1.0F;
  auto & inside = upper.motor_cmd[22];  // every field inside its limit
  inside.q = 0.25F;
  inside.dq = 1.5F;
  inside.tau = -2.5F;
  inside.kp = 40.0F;
  inside.kd = 3.0F;
  const auto inside_sent = inside;
  LowCmd lower = command(h12());
  lower.motor_cmd[3].q = 10.0F;
  lower.motor_cmd[3].kp = 600.0F;

  core.on_command(Input::Lower, lower, "lower", 0.001);
  core.on_command(Input::Upper, upper, "upper", 0.001);
  const auto out = core.step(0.002);
  ASSERT_TRUE(out);

  const auto & o20 = out->motor_cmd[20];
  EXPECT_EQ(o20.q, static_cast<float>(limits.q_clip_high[20]));
  EXPECT_EQ(o20.dq, static_cast<float>(limits.dq_clip[20]));
  EXPECT_EQ(o20.tau, static_cast<float>(-limits.tau_clip[20]));
  EXPECT_EQ(o20.kp, static_cast<float>(limits.kp_max));
  EXPECT_EQ(o20.kd, static_cast<float>(limits.kd_max));
  const auto & o21 = out->motor_cmd[21];
  EXPECT_EQ(o21.q, static_cast<float>(limits.q_clip_low[21]));
  EXPECT_EQ(o21.dq, static_cast<float>(-limits.dq_clip[21]));
  EXPECT_EQ(o21.tau, static_cast<float>(limits.tau_clip[21]));
  EXPECT_EQ(o21.kp, 0.0F);
  EXPECT_EQ(o21.kd, 0.0F);
  EXPECT_EQ(out->motor_cmd[22], inside_sent);
  EXPECT_EQ(out->motor_cmd[3].q, static_cast<float>(limits.q_clip_high[3]));
  EXPECT_EQ(out->motor_cmd[3].kp, static_cast<float>(limits.kp_max));

  // A gain above its maximum is reported once per joint and gain: kp and kd on
  // right_shoulder_pitch_joint, kp on left_knee_joint.
  auto events = core.drain_events();
  EXPECT_EQ(count(events, Event::Warn), 3U) << texts(events);
  EXPECT_TRUE(has_event(events, Event::Warn, "right_shoulder_pitch_joint")) << texts(events);
  EXPECT_TRUE(has_event(events, Event::Warn, "left_knee_joint")) << texts(events);
  core.on_command(Input::Lower, lower, "lower", 0.003);
  core.on_command(Input::Upper, upper, "upper", 0.003);
  ASSERT_TRUE(core.step(0.004));
  events = core.drain_events();
  EXPECT_EQ(count(events, Event::Warn), 0U) << texts(events);
  EXPECT_FALSE(core.estopped());
}

TEST(Clip, NonFiniteSampleLatches)
{
  // H1-2 split, engaged: a NaN q on an upper slot latches, and its burst follows.
  H12Split rig;
  ASSERT_TRUE(rig.core.step(0.0));
  LowCmd bad = command(h12());
  bad.motor_cmd[20].q = std::numeric_limits<float>::quiet_NaN();
  rig.core.on_command(Input::Upper, bad, "A", 0.002);
  ASSERT_TRUE(rig.core.estopped());
  EXPECT_EQ(
    rig.core.estop_reason(),
    "Command validation failed on /safety/lowcmd_upper_in: "
    "motor 20 (right_shoulder_pitch_joint) q is nan");
  const auto out = rig.core.step(0.004);
  ASSERT_TRUE(out);
  expect_same(*out, golem_safety::estop_command(h12(), rig.cfg, *out));

  // G1 split, engaged: the same, then the estop command on every step.
  const SafetyConfig g1_cfg = config(g1(), "sim_split");
  SafetyCore g1_core(g1(), g1_cfg, false, kDomain);
  g1_core.on_state(state_of(g1()), 0.0);
  g1_core.on_command(Input::Lower, command(g1()), "lower", 0.001);
  g1_core.on_command(Input::Upper, command(g1()), "upper", 0.001);
  ASSERT_TRUE(g1_core.step(0.002));
  LowCmd g1_bad = command(g1());
  g1_bad.motor_cmd[20].q = std::numeric_limits<float>::quiet_NaN();
  g1_core.on_command(Input::Upper, g1_bad, "upper", 0.003);
  ASSERT_TRUE(g1_core.estopped());
  EXPECT_TRUE(contains(g1_core.estop_reason(), "/safety/lowcmd_upper_in")) <<
    g1_core.estop_reason();
  EXPECT_TRUE(contains(g1_core.estop_reason(), "left_wrist_pitch_joint")) <<
    g1_core.estop_reason();
  const auto g1_out = g1_core.step(0.004);
  ASSERT_TRUE(g1_out);
  expect_same(*g1_out, golem_safety::estop_command(g1(), g1_cfg, *g1_out));

  // Every field, NaN and either infinity, on both robots and every input; a body slot the
  // input does not own counts too.
  const float values[] = {
    std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
    Input input;
    const char * topic;
    int slot;
  };
  const std::vector<Case> cases = {
    {h12(), "sim_safety_split", Input::Upper, "/safety/lowcmd_upper_in", 13},
    {h12(), "sim_safety_split", Input::Lower, "/safety/lowcmd_lower_in", 4},
    {h12(), "sim_safety_split", Input::Lower, "/safety/lowcmd_lower_in", 26},
    {h12(), "default_safety_full", Input::Full, "/safety/lowcmd_in", 26},
    {g1(), "sim_split", Input::Upper, "/safety/lowcmd_upper_in", 3},
    {g1(), "sim_arm_sdk", Input::Upper, "/safety/lowcmd_upper_in", 28},
    {g1(), "full", Input::Full, "/safety/lowcmd_in", 28},
  };
  for (const Case & c : cases) {
    for (const char * field : {"q", "dq", "tau", "kp", "kd"}) {
      for (const float value : values) {
        SCOPED_TRACE(
          std::string(c.config_name) + " " + c.topic + " slot " + std::to_string(c.slot) +
          " " + field + " = " + std::to_string(value));
        SafetyCore core(c.profile, config(c.profile, c.config_name), false, kDomain);
        core.on_state(state_of(c.profile), 0.0);
        LowCmd broken = command(c.profile);
        field_of(broken.motor_cmd[c.slot], field) = value;
        core.on_command(c.input, broken, "writer", 0.001);
        ASSERT_TRUE(core.estopped());
        const std::string & reason = core.estop_reason();
        EXPECT_TRUE(contains(reason, c.topic)) << reason;
        EXPECT_TRUE(contains(reason, c.profile.joints[c.slot].name)) << reason;
        EXPECT_TRUE(contains(reason, std::string(") ") + field + " is ")) << reason;
      }
    }
  }
}

TEST(Clip, NonFiniteFirstSampleLatches)
{
  // zero_stream: a first command that is not valid latches before any input engages; the
  // burst is the estop command.
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  SafetyCore core(h12(), cfg, false, kDomain);
  core.on_state(state_of(h12()), 0.0);
  ASSERT_TRUE(core.step(0.0));
  LowCmd bad = command(h12(), 0.1F);
  bad.motor_cmd[15].kd = std::numeric_limits<float>::quiet_NaN();
  core.on_command(Input::Upper, bad, "upper", 0.001);
  ASSERT_TRUE(core.estopped());
  EXPECT_TRUE(contains(core.estop_reason(), "left_shoulder_yaw_joint")) << core.estop_reason();
  const auto out = core.step(0.002);
  ASSERT_TRUE(out);
  expect_same(*out, golem_safety::estop_command(h12(), cfg, *out));
  // A valid command after the latch changes nothing.
  core.on_command(Input::Upper, command(h12(), 0.1F), "upper", 0.003);
  expect_same(*core.step(0.004), golem_safety::estop_command(h12(), cfg, *out));

  // engage_gate: the latch comes before engaging, so nothing is ever written.
  SafetyCore arm_sdk(g1(), config(g1(), "sim_arm_sdk"), false, kDomain);
  arm_sdk.on_state(state_of(g1()), 0.0);
  LowCmd g1_bad = command(g1());
  g1_bad.motor_cmd[20].dq = std::numeric_limits<float>::infinity();
  arm_sdk.on_command(Input::Upper, g1_bad, "upper", 0.001);
  ASSERT_TRUE(arm_sdk.estopped());
  arm_sdk.on_command(Input::Upper, command(g1()), "upper", 0.002);
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(arm_sdk.step(0.003 + i * kDt));
  }
  const auto events = arm_sdk.drain_events();
  EXPECT_TRUE(has_event(events, Event::Error, "estop_latched_silent")) << texts(events);
}

TEST(Limits, MeasuredPastEstopLatches)
{
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  const auto & limits = cfg.limits;

  {
    SafetyCore core(h12(), cfg, false, kDomain);
    core.on_state(state_of(h12()), 0.0);
    // Past the clip range but inside the estop range: no latch.
    LowState inside = state_of(h12());
    inside.motor_state[20].q = static_cast<float>(limits.q_clip_high[20] + 0.05);
    core.on_state(inside, 0.002);
    EXPECT_FALSE(core.estopped());

    LowState state = state_of(h12());
    state.motor_state[20].q = static_cast<float>(limits.q_estop_high[20] + 0.2);
    core.on_state(state, 0.004);
    ASSERT_TRUE(core.estopped());
    const std::string & reason = core.estop_reason();
    EXPECT_TRUE(contains(reason, "right_shoulder_pitch_joint")) << reason;
    EXPECT_TRUE(contains(reason, "x")) << reason;
    EXPECT_TRUE(contains(reason, " q ")) << reason;
    EXPECT_TRUE(contains(reason, "limit 1.670")) << reason;
    EXPECT_TRUE(contains(reason, "1.12x")) << reason;
  }

  struct Case
  {
    const char * what;
    int slot;
    float q, dq, tau;
    std::vector<std::string> parts;
  };
  const std::vector<Case> cases = {
    {"q below", 3, static_cast<float>(limits.q_estop_low[3] - 0.1), 0.0F, 0.0F,
      {"left_knee_joint", " q ", "x)"}},
    {"dq", 3, 0.0F, static_cast<float>(-2.0 * limits.dq_estop[3]), 0.0F,
      {"left_knee_joint", " dq ", "2.00x"}},
    {"tau", 12, 0.0F, 0.0F, static_cast<float>(1.5 * limits.tau_estop[12]),
      {"torso_joint", " tau ", "1.50x"}},
    {"non-finite", 0, std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F,
      {"left_hip_yaw_joint", "nan"}},
  };
  for (const Case & c : cases) {
    SCOPED_TRACE(c.what);
    SafetyCore core(h12(), cfg, false, kDomain);
    LowState state = state_of(h12());
    state.motor_state[c.slot].q = c.q;
    state.motor_state[c.slot].dq = c.dq;
    state.motor_state[c.slot].tau_est = c.tau;
    core.on_state(state, 0.0);
    ASSERT_TRUE(core.estopped());
    for (const auto & part : c.parts) {
      EXPECT_TRUE(contains(core.estop_reason(), part)) << core.estop_reason();
    }
  }
}

TEST(Limits, EngageGateChecksOnlyEngagedSlotsOnceEngaged)
{
  const SafetyConfig cfg = config(g1(), "sim_arm_sdk");
  SafetyCore core(g1(), cfg, false, kDomain);
  // Before engaging a joint past its estop range is not ours to guard.
  core.on_state(past_estop(g1(), cfg, 20), 0.0);
  EXPECT_FALSE(core.estopped());
  core.on_state(state_of(g1()), 0.01);
  core.on_command(Input::Upper, command(g1()), "upper", 0.02);
  ASSERT_TRUE(core.step(0.02));
  // arm_sdk_mode guards the waist and arms only: the legs are Unitree's.
  core.on_state(past_estop(g1(), cfg, 3), 0.03);
  EXPECT_FALSE(core.estopped());
  core.on_state(past_estop(g1(), cfg, 20), 0.04);
  ASSERT_TRUE(core.estopped());
  EXPECT_TRUE(contains(core.estop_reason(), "left_wrist_pitch_joint")) << core.estop_reason();
}

TEST(Identity, ForeignSampleAfterEngageLatches)
{
  SafetyCore core(h12(), config(h12(), "sim_safety_split"), false, kDomain);
  core.on_state(state_of(h12()), 0.0);
  LowState foreign = state_of(h12());
  foreign.mode_machine = 5;
  core.on_state(foreign, 0.001);
  EXPECT_FALSE(core.estopped()) << "before engaging a foreign sample is only recorded";

  core.on_state(state_of(h12()), 0.002);
  core.on_command(Input::Lower, command(h12()), "lower", 0.003);
  core.on_command(Input::Upper, command(h12()), "upper", 0.003);
  ASSERT_TRUE(core.step(0.004));
  EXPECT_FALSE(core.estopped());

  core.on_state(foreign, 0.005);
  ASSERT_TRUE(core.estopped());
  EXPECT_TRUE(contains(core.estop_reason(), "mode_machine 5")) << core.estop_reason();
  EXPECT_TRUE(contains(core.estop_reason(), "6")) << core.estop_reason();

  // The G1 profile, engaged, against an H1-2 sample.
  SafetyCore g1_core(g1(), config(g1(), "sim_split"), false, kDomain);
  g1_core.on_state(state_of(g1()), 0.0);
  g1_core.on_command(Input::Lower, command(g1()), "lower", 0.001);
  g1_core.on_command(Input::Upper, command(g1()), "upper", 0.001);
  ASSERT_TRUE(g1_core.step(0.002));
  LowState h12_sample = state_of(g1());
  h12_sample.mode_machine = 6;
  g1_core.on_state(h12_sample, 0.003);
  EXPECT_TRUE(g1_core.estopped());
}

TEST(SingleWriter, TwoLiveWritersLatch)
{
  H12Split rig;
  LowCmd from_a = command(h12());
  from_a.motor_cmd[20].q = 0.3F;
  for (int i = 1; i <= 100; ++i) {
    const double t = i * 0.01;
    rig.core.on_state(state_of(h12()), t);
    rig.core.on_command(Input::Lower, command(h12()), "lower", t);
    rig.core.on_command(Input::Upper, from_a, "A", t);
    ASSERT_TRUE(rig.core.step(t));
  }
  // B 0.2 s after A's last command (t = 1.0).
  LowCmd from_b = command(h12());
  from_b.motor_cmd[20].q = 0.4F;
  rig.core.on_state(state_of(h12()), 1.2);
  rig.core.on_command(Input::Upper, from_b, "B", 1.2);
  ASSERT_TRUE(rig.core.estopped());
  const std::string & reason = rig.core.estop_reason();
  EXPECT_EQ(reason.rfind("second writer on /safety/lowcmd_upper_in", 0), 0U) << reason;
  const auto out = rig.core.step(1.2);
  ASSERT_TRUE(out);
  expect_same(*out, golem_safety::estop_command(h12(), rig.cfg, *out));

  // A gap of exactly handover_seconds is not a handover.
  H12Split edge;
  edge.core.on_command(Input::Upper, from_a, "A", 1.0);
  edge.core.on_command(Input::Upper, from_b, "B", 1.5);
  EXPECT_TRUE(edge.core.estopped());

  // Nor is a second writer on the lower input.
  H12Split lower;
  lower.core.on_command(Input::Lower, command(h12()), "other", 0.1);
  ASSERT_TRUE(lower.core.estopped());
  EXPECT_EQ(
    lower.core.estop_reason().rfind("second writer on /safety/lowcmd_lower_in", 0), 0U) <<
    lower.core.estop_reason();
}

TEST(SingleWriter, RestartAfterQuietIsAccepted)
{
  H12Split rig;
  LowCmd from_a = command(h12());
  from_a.motor_cmd[20].q = 0.3F;
  for (int i = 1; i <= 159; ++i) {
    const double t = i * 0.01;
    rig.core.on_state(state_of(h12()), t);
    rig.core.on_command(Input::Lower, command(h12()), "lower", t);
    if (i <= 100) {
      rig.core.on_command(Input::Upper, from_a, "A", t);
    }
    ASSERT_TRUE(rig.core.step(t));
  }
  rig.core.drain_events();

  // A has been quiet for 0.6 s (> handover_seconds 0.5): B takes the input over.
  LowCmd from_b = command(h12());
  from_b.motor_cmd[20].q = 0.4F;
  rig.core.on_state(state_of(h12()), 1.6);
  rig.core.on_command(Input::Lower, command(h12()), "lower", 1.6);
  rig.core.on_command(Input::Upper, from_b, "B", 1.6);
  const auto out = rig.core.step(1.6);
  EXPECT_FALSE(rig.core.estopped()) << rig.core.estop_reason();
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd[20].q, 0.4F);
  const auto events = rig.core.drain_events();
  EXPECT_EQ(count(events, Event::Info), 1U) << texts(events);
  EXPECT_EQ(count(events, Event::Warn), 0U) << texts(events);
  EXPECT_EQ(count(events, Event::Error), 0U) << texts(events);
  EXPECT_TRUE(has_event(events, Event::Info, "/safety/lowcmd_upper_in")) << texts(events);

  // B is now the current writer: A coming back while B is live latches.
  rig.core.on_command(Input::Upper, from_a, "A", 1.61);
  EXPECT_TRUE(rig.core.estopped());
}

TEST(Estop, BurstThenSilence)
{
  H12Split rig;
  ASSERT_TRUE(rig.core.step(0.0));
  rig.core.on_state(past_estop(h12(), rig.cfg, 20), 0.002);
  ASSERT_TRUE(rig.core.estopped());
  const LowCmd expected =
    golem_safety::estop_command(h12(), rig.cfg, golem_safety::zero_command(h12()));

  double t = 0.004;
  for (int i = 0; i < 50; ++i, t += kDt) {
    SCOPED_TRACE("estop command " + std::to_string(i));
    // Commands after the latch change nothing.
    rig.core.on_command(Input::Upper, command(h12(), 0.2F), "A", t);
    const auto out = rig.core.step(t);
    ASSERT_TRUE(out);
    expect_same(*out, expected);
  }
  for (int i = 0; i < 1000; ++i, t += kDt) {
    ASSERT_FALSE(rig.core.step(t)) << "step " << i << " after the burst";
  }

  const auto events = rig.core.drain_events();
  EXPECT_TRUE(has_event(events, Event::Error, "\"event\": \"estop_command_written\"")) <<
    texts(events);
  EXPECT_TRUE(has_event(events, Event::Error, "\"decision_ms\": 2,")) << texts(events);
  EXPECT_TRUE(has_event(events, Event::Error, "\"action_ms\": 4,")) << texts(events);
  EXPECT_TRUE(has_event(events, Event::Error, "\"decision_to_action_ms\": 2,")) << texts(events);
  EXPECT_TRUE(has_event(events, Event::Error, "right_shoulder_pitch_joint")) << texts(events);
}

TEST(Estop, KeepWriting)
{
  const SafetyConfig cfg = config(g1(), "sim_split");
  SafetyCore core(g1(), cfg, false, kDomain);
  core.on_state(state_of(g1()), 0.0);
  core.on_command(Input::Lower, command(g1()), "lower", 0.001);
  core.on_command(Input::Upper, command(g1()), "upper", 0.001);
  ASSERT_TRUE(core.step(0.002));
  core.on_state(past_estop(g1(), cfg, 20), 0.003);
  ASSERT_TRUE(core.estopped());

  const LowCmd expected =
    golem_safety::estop_command(g1(), cfg, golem_safety::zero_command(g1()));
  double t = 0.004;
  for (int i = 0; i < 1000; ++i, t += kDt) {
    const auto out = core.step(t);
    ASSERT_TRUE(out) << "step " << i;
    expect_same(*out, expected);
  }
}

TEST(Estop, SilentWhenNeverEngaged)
{
  // engage_gate: a latch before engaging writes nothing, then or later.
  SafetyCore core(g1(), config(g1(), "sim_split"), false, kDomain);
  core.on_state(state_of(g1()), 0.0);
  core.on_command(Input::Lower, command(g1()), "lower", 0.001);
  core.on_command(Input::Lower, command(g1()), "other", 0.002);
  ASSERT_TRUE(core.estopped());
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(core.step(0.003 + i * kDt));
  }
  const auto events = core.drain_events();
  EXPECT_TRUE(has_event(events, Event::Error, "estop_latched_silent")) << texts(events);
}

TEST(Estop, ArmSdkReleaseRamp)
{
  const SafetyConfig cfg = config(g1(), "sim_arm_sdk");
  ASSERT_EQ(cfg.arm_sdk.release_seconds, 1.0);
  const LowCmd zero = golem_safety::zero_command(g1());
  SafetyCore core(g1(), cfg, false, kDomain);
  LowCmd upper = command(g1());
  upper.motor_cmd[20].q = 0.05F;
  const LowCmd last = engage_arm_sdk(core, upper, 1.5);
  ASSERT_EQ(last.motor_cmd[29].q, 1.0F);

  core.on_state(past_estop(g1(), cfg, 20), 1.505);
  ASSERT_TRUE(core.estopped());
  const double start = 1.51;
  for (int i = 0; i <= 300; ++i) {
    const double t = start + i * 0.01;
    SCOPED_TRACE("t = " + std::to_string(t));
    core.on_state(state_of(g1()), t);
    const auto out = core.step(t);
    ASSERT_TRUE(out);
    const double expected = std::max(0.0, 1.0 - (t - start) / cfg.arm_sdk.release_seconds);
    EXPECT_NEAR(out->motor_cmd[29].q, expected, 1e-6);
    if (t > start + cfg.arm_sdk.release_seconds + 1e-9) {
      EXPECT_EQ(out->motor_cmd[29].q, 0.0F);
    }
    for (int slot = 12; slot <= 28; ++slot) {
      ASSERT_EQ(out->motor_cmd[slot], last.motor_cmd[slot]) << "slot " << slot;
    }
    for (int slot = 0; slot <= 11; ++slot) {
      ASSERT_EQ(out->motor_cmd[slot], zero.motor_cmd[slot]) << "slot " << slot;
    }
    expect_stamped(*out, g1());
  }
  const auto events = core.drain_events();
  EXPECT_TRUE(has_event(events, Event::Error, "estop_command_written")) << texts(events);
}

TEST(Estop, ArmSdkReleaseWaitsForLowstate)
{
  SafetyConfig cfg = config(g1(), "sim_arm_sdk");
  cfg.stale.state_seconds = 0.1;
  SafetyCore core(g1(), cfg, false, kDomain);
  engage_arm_sdk(core, command(g1()), 1.5);
  core.on_state(past_estop(g1(), cfg, 20), 1.505);
  ASSERT_TRUE(core.estopped());
  core.drain_events();

  // /lowstate stops: the ramp runs while the last one (t = 1.505) is at most 0.1 s old,
  // then holds.
  float held = 0.0F;
  for (int i = 1; i <= 50; ++i) {
    const double t = 1.5 + i * 0.01;
    const auto out = core.step(t);
    ASSERT_TRUE(out);
    if (i <= 10) {
      EXPECT_NEAR(out->motor_cmd[29].q, 1.0 - (t - 1.51), 1e-6) << "t = " << t;
      held = out->motor_cmd[29].q;
    } else {
      EXPECT_EQ(out->motor_cmd[29].q, held) << "t = " << t;
    }
  }
  EXPECT_NEAR(held, 0.91, 1e-6);
  auto events = core.drain_events();
  EXPECT_TRUE(has_event(events, Event::Warn, "held")) << texts(events);

  // /lowstate flows again: the ramp goes on from where it held, within one step.
  core.on_state(state_of(g1()), 2.0);
  auto out = core.step(2.0);
  ASSERT_TRUE(out);
  const float resumed = out->motor_cmd[29].q;
  EXPECT_NEAR(resumed, held, 0.0101);
  EXPECT_LE(resumed, held);
  core.on_state(state_of(g1()), 2.1);
  out = core.step(2.1);
  ASSERT_TRUE(out);
  EXPECT_NEAR(out->motor_cmd[29].q, resumed - 0.1, 1e-6);
  events = core.drain_events();
  EXPECT_TRUE(has_event(events, Event::Info, "arm_sdk_release_ramping")) << texts(events);
}

TEST(EstopInput, TriggeredOrUnpluggedLatchesWithUseEstop)
{
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  {
    SafetyCore core(h12(), cfg, true, kDomain);
    core.on_estop(false, true, 0.0);
    EXPECT_FALSE(core.estopped());
    core.on_estop(true, true, 0.010);
    ASSERT_TRUE(core.estopped());
    EXPECT_EQ(core.estop_reason(), "External estop triggered");
    ASSERT_TRUE(core.step(0.012));
    const auto events = core.drain_events();
    EXPECT_TRUE(has_event(events, Event::Error, "\"trigger_stamp_ms\": 10,")) << texts(events);
    EXPECT_TRUE(has_event(events, Event::Error, "\"trigger_to_action_ms\": 2")) << texts(events);
  }
  {
    SafetyCore core(h12(), cfg, true, kDomain);
    core.on_estop(false, false, 0.0);
    ASSERT_TRUE(core.estopped());
    EXPECT_EQ(core.estop_reason(), "External estop unplugged");
  }
  {
    SafetyCore core(h12(), cfg, false, kDomain);
    core.on_estop(true, false, 0.0);
    EXPECT_FALSE(core.estopped());
  }
}

TEST(ExitStep, FullAndSplitAreDoneAtOnce)
{
  H12Split rig;
  ASSERT_TRUE(rig.core.step(0.0));
  EXPECT_FALSE(rig.core.exit_step(0.002));

  // Estopped mid-burst, the burst is finished first.
  H12Split burst;
  ASSERT_TRUE(burst.core.step(0.0));
  burst.core.on_state(past_estop(h12(), burst.cfg, 20), 0.001);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(burst.core.step(0.002 + i * kDt));
  }
  for (int i = 10; i < 50; ++i) {
    ASSERT_TRUE(burst.core.exit_step(0.022 + i * kDt)) << i;
  }
  EXPECT_FALSE(burst.core.exit_step(0.2));

  // keep_writing: the estop command once if none was written since the latch, then done.
  const SafetyConfig g1_cfg = config(g1(), "sim_split");
  for (const bool stepped : {false, true}) {
    SCOPED_TRACE(stepped ? "a step after the latch" : "no step after the latch");
    SafetyCore keep(g1(), g1_cfg, false, kDomain);
    keep.on_state(state_of(g1()), 0.0);
    keep.on_command(Input::Lower, command(g1()), "lower", 0.001);
    keep.on_command(Input::Upper, command(g1()), "upper", 0.001);
    ASSERT_TRUE(keep.step(0.002));
    keep.on_state(past_estop(g1(), g1_cfg, 20), 0.003);
    ASSERT_TRUE(keep.estopped());
    if (stepped) {
      ASSERT_TRUE(keep.step(0.004));
    } else {
      const auto last = keep.exit_step(0.004);
      ASSERT_TRUE(last);
      expect_same(*last, golem_safety::estop_command(g1(), g1_cfg, *last));
    }
    EXPECT_FALSE(keep.exit_step(0.006));
    EXPECT_FALSE(keep.step(0.008));
  }
}

TEST(ExitStep, ArmSdkRampsWeightToZero)
{
  const SafetyConfig cfg = config(g1(), "sim_arm_sdk");
  SafetyCore core(g1(), cfg, false, kDomain);
  const LowCmd last = engage_arm_sdk(core, command(g1()), 1.5);
  ASSERT_EQ(last.motor_cmd[29].q, 1.0F);

  // No /lowstate during the exit: the ramp runs anyway.
  const double start = 1.51;
  std::optional<LowCmd> out;
  int written = 0;
  for (int i = 0; i < 1000; ++i) {
    out = core.exit_step(start + i * 0.01);
    if (!out) {
      break;
    }
    ++written;
    const double t = start + i * 0.01;
    EXPECT_NEAR(
      out->motor_cmd[29].q, std::max(0.0, 1.0 - (t - start) / cfg.arm_sdk.release_seconds),
      1e-6);
    EXPECT_EQ(out->motor_cmd[20], last.motor_cmd[20]);
  }
  EXPECT_FALSE(out);
  EXPECT_GE(written, 101);
  EXPECT_LE(written, 102);
  EXPECT_FALSE(core.estopped());

  // Never engaged: done at once.
  SafetyCore idle(g1(), cfg, false, kDomain);
  idle.on_state(state_of(g1()), 0.0);
  EXPECT_FALSE(idle.exit_step(0.01));
}

TEST(Log, OneSamplePerLowstateOnTheBodySlots)
{
  SafetyConfig cfg = config(h12(), "sim_safety_split");
  {
    SafetyCore core(h12(), cfg, false, kDomain);
    core.on_state(state_of(h12()), 0.0);
    EXPECT_FALSE(core.take_log_sample()) << "logging is off in this config";
  }
  cfg.logging.enabled = true;
  SafetyCore core(h12(), cfg, false, kDomain);
  LowState state = state_of(h12(), 0.1F);
  state.motor_state[4].temperature = {31, 32};
  state.imu_state.quaternion = {1.0F, 0.0F, 0.0F, 0.0F};
  core.on_state(state, 0.0);
  LowCmd upper = command(h12());
  upper.motor_cmd[20].q = 0.25F;
  core.on_command(Input::Lower, command(h12()), "lower", 0.001);
  core.on_command(Input::Upper, upper, "upper", 0.001);
  ASSERT_TRUE(core.step(0.002));

  const auto sample = core.take_log_sample();
  ASSERT_TRUE(sample);
  EXPECT_EQ(sample->q.size(), 27U);
  EXPECT_EQ(sample->q_cmd.size(), 27U);
  EXPECT_EQ(sample->q[26], 0.1F);
  EXPECT_EQ(sample->temperature[4][1], 32);
  EXPECT_EQ(sample->imu_quaternion[0], 1.0F);
  EXPECT_EQ(sample->q_cmd[20], 0.25F);
  EXPECT_EQ(sample->kp_cmd[20], 60.0F);
  EXPECT_GT(sample->time_stamp, 1.0e9) << "wall clock, s";
  EXPECT_FALSE(core.take_log_sample()) << "one sample per /lowstate";

  // The latching /lowstate is logged; none after it.
  core.on_state(past_estop(h12(), cfg, 20), 0.004);
  ASSERT_TRUE(core.estopped());
  EXPECT_TRUE(core.take_log_sample());
  core.on_state(state_of(h12()), 0.006);
  EXPECT_FALSE(core.take_log_sample());

  // The G1 logs its 29 body slots.
  SafetyCore g1_core(g1(), config(g1(), "sim_split"), false, kDomain);
  g1_core.on_state(state_of(g1()), 0.0);
  const auto g1_sample = g1_core.take_log_sample();
  ASSERT_TRUE(g1_sample);
  EXPECT_EQ(g1_sample->q.size(), 29U);
  EXPECT_EQ(g1_sample->kd_cmd.size(), 29U);
}

TEST(Events, GainClippedIsTheOldJson)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
    Input input;
    int slot;
    const char * gain;
    float commanded;
    std::string text;
  };
  const std::vector<Case> cases = {
    // the old G1 test's case (test_hardening_dds.py)
    {g1(), "sim_arm_sdk", Input::Upper, 15, "kp", 300.0F,
      "{\"commanded\": 300.0, \"event\": \"gain_clipped\", \"gain\": \"kp\", "
      "\"joint\": \"left_shoulder_pitch_joint\", \"max\": 200.0, \"source\": \"arm_sdk_mode\"}"},
    {g1(), "sim_split", Input::Upper, 20, "kp", 300.1F,
      "{\"commanded\": 300.1000061035156, \"event\": \"gain_clipped\", \"gain\": \"kp\", "
      "\"joint\": \"left_wrist_pitch_joint\", \"max\": 200.0, \"source\": \"split_mode_upper\"}"},
    {h12(), "sim_safety_split", Input::Lower, 3, "kd", 31.5F,
      "{\"commanded\": 31.5, \"event\": \"gain_clipped\", \"gain\": \"kd\", "
      "\"joint\": \"left_knee_joint\", \"max\": 30.0, \"source\": \"split_mode_lower\"}"},
    {h12(), "sim_safety_split", Input::Upper, 20, "kd", 1e20F,
      "{\"commanded\": 1.0000000200408773e+20, \"event\": \"gain_clipped\", \"gain\": \"kd\", "
      "\"joint\": \"right_shoulder_pitch_joint\", \"max\": 30.0, "
      "\"source\": \"split_mode_upper\"}"},
    {h12(), "default_safety_full", Input::Full, 26, "kp", 600.0F,
      "{\"commanded\": 600.0, \"event\": \"gain_clipped\", \"gain\": \"kp\", "
      "\"joint\": \"right_wrist_yaw_joint\", \"max\": 500.0, \"source\": \"full_body_mode\"}"},
  };
  for (const Case & c : cases) {
    SCOPED_TRACE(c.text);
    SafetyCore core(c.profile, config(c.profile, c.config_name), false, kDomain);
    core.on_state(state_of(c.profile), 0.0);
    LowCmd cmd = command(c.profile);
    field_of(cmd.motor_cmd[c.slot], c.gain) = c.commanded;
    for (int i = 0; i < 5; ++i) {
      core.on_command(c.input, cmd, "writer", 0.001 * i);
    }
    const auto clipped = named(core.drain_events(), "gain_clipped");
    ASSERT_EQ(clipped.size(), 1U) << "once per joint and gain";
    EXPECT_EQ(clipped[0].first.kind, Event::Warn);
    EXPECT_EQ(clipped[0].first.text, c.text);
    const YAML::Node & e = clipped[0].second;
    EXPECT_EQ(
      keys_of(e),
      (std::vector<std::string>{"commanded", "event", "gain", "joint", "max", "source"}));
    EXPECT_EQ(e["commanded"].as<double>(), static_cast<double>(c.commanded));
    EXPECT_EQ(e["gain"].as<std::string>(), c.gain);
    EXPECT_EQ(e["joint"].as<std::string>(), c.profile.joints[c.slot].name);
  }
}

TEST(Events, ArmSdkReleaseEventsAreTheOldJson)
{
  SafetyConfig cfg = config(g1(), "sim_arm_sdk");
  cfg.stale.state_seconds = 0.1;
  SafetyCore core(g1(), cfg, false, kDomain);
  engage_arm_sdk(core, command(g1()), 1.5);
  core.on_state(past_estop(g1(), cfg, 20), 1.505);
  ASSERT_TRUE(core.estopped());
  core.drain_events();

  // The ramp runs from t = 1.51, holds once /lowstate is over 0.1 s old, and runs again
  // when it flows; the estop report comes first, as the old layers logged it.
  for (int i = 1; i <= 50; ++i) {
    ASSERT_TRUE(core.step(1.5 + i * 0.01));
  }
  auto events = core.drain_events();
  std::vector<std::string> sequence;
  for (const auto & event : events) {
    if (!event.text.empty() && event.text[0] == '{') {
      sequence.push_back(YAML::Load(event.text)["event"].as<std::string>());
    }
  }
  EXPECT_EQ(
    sequence, (std::vector<std::string>{
    "estop_command_written", "arm_sdk_release_ramping", "arm_sdk_release_held"})) <<
    texts(events);
  const auto ramping = named(events, "arm_sdk_release_ramping");
  const auto held = named(events, "arm_sdk_release_held");
  ASSERT_EQ(ramping.size(), 1U);
  ASSERT_EQ(held.size(), 1U);
  EXPECT_EQ(ramping[0].first.kind, Event::Info);
  EXPECT_EQ(
    ramping[0].first.text,
    "{\"event\": \"arm_sdk_release_ramping\", \"state_age\": 0.005, \"weight\": 1.0}");
  EXPECT_EQ(held[0].first.kind, Event::Warn);
  EXPECT_EQ(
    held[0].first.text,
    "{\"event\": \"arm_sdk_release_held\", \"state_age\": 0.105, \"weight\": 0.91}");
  for (const YAML::Node * node : {&ramping[0].second, &held[0].second}) {
    EXPECT_EQ(keys_of(*node), (std::vector<std::string>{"event", "state_age", "weight"}));
  }

  core.on_state(state_of(g1()), 2.0);
  ASSERT_TRUE(core.step(2.0));
  const auto resumed = named(core.drain_events(), "arm_sdk_release_ramping");
  ASSERT_EQ(resumed.size(), 1U);
  EXPECT_EQ(
    resumed[0].first.text,
    "{\"event\": \"arm_sdk_release_ramping\", \"state_age\": 0.0, \"weight\": 0.91}");
  EXPECT_EQ(resumed[0].second["weight"].as<double>(), 0.91);
}

TEST(Events, EstopReportIsTheOldJsonAfterTheCommand)
{
  // Built only once the step after the latch has handed its command back.
  H12Split rig;
  ASSERT_TRUE(rig.core.step(0.0));
  rig.core.on_state(past_estop(h12(), rig.cfg, 20), 0.002);
  ASSERT_TRUE(rig.core.estopped());
  EXPECT_TRUE(named(rig.core.drain_events(), "estop_command_written").empty());
  ASSERT_TRUE(rig.core.step(0.004));
  auto reports = named(rig.core.drain_events(), "estop_command_written");
  ASSERT_EQ(reports.size(), 1U);
  EXPECT_EQ(reports[0].first.kind, Event::Error);
  const YAML::Node & written = reports[0].second;
  EXPECT_EQ(
    keys_of(written), (std::vector<std::string>{
    "action_ms", "decision_ms", "decision_to_action_ms", "event", "reason",
    "trigger_stamp_ms"}));
  EXPECT_EQ(written["action_ms"].as<long long>(), 4);
  EXPECT_EQ(written["decision_ms"].as<long long>(), 2);
  EXPECT_EQ(written["decision_to_action_ms"].as<long long>(), 2);
  EXPECT_EQ(written["reason"].as<std::string>(), rig.core.estop_reason());
  EXPECT_TRUE(written["trigger_stamp_ms"].IsNull());
  ASSERT_TRUE(rig.core.step(0.006));
  EXPECT_TRUE(named(rig.core.drain_events(), "estop_command_written").empty()) << "once";

  // The hardware estop adds trigger_to_action_ms.
  SafetyCore box(h12(), rig.cfg, true, kDomain);
  box.on_estop(true, true, 0.010);
  ASSERT_TRUE(box.step(0.013));
  reports = named(box.drain_events(), "estop_command_written");
  ASSERT_EQ(reports.size(), 1U);
  EXPECT_EQ(
    keys_of(reports[0].second), (std::vector<std::string>{
    "action_ms", "decision_ms", "decision_to_action_ms", "event", "reason",
    "trigger_stamp_ms", "trigger_to_action_ms"}));
  EXPECT_EQ(reports[0].second["trigger_stamp_ms"].as<long long>(), 10);
  EXPECT_EQ(reports[0].second["trigger_to_action_ms"].as<long long>(), 3);
  EXPECT_EQ(reports[0].second["reason"].as<std::string>(), "External estop triggered");

  // Nothing written: estop_latched_silent.
  SafetyCore silent(g1(), config(g1(), "sim_split"), false, kDomain);
  silent.on_state(state_of(g1()), 0.0);
  silent.on_command(Input::Lower, command(g1()), "lower", 0.001);
  silent.on_command(Input::Lower, command(g1()), "other", 0.002);
  ASSERT_TRUE(silent.estopped());
  EXPECT_FALSE(silent.step(0.004));
  const auto silences = named(silent.drain_events(), "estop_latched_silent");
  ASSERT_EQ(silences.size(), 1U);
  EXPECT_EQ(
    keys_of(silences[0].second),
    (std::vector<std::string>{"decision_ms", "event", "reason", "trigger_stamp_ms"}));
  EXPECT_EQ(silences[0].second["decision_ms"].as<long long>(), 2);
  EXPECT_EQ(silences[0].second["reason"].as<std::string>(), silent.estop_reason());
}

TEST(Construction, RefusesAConfigForAnotherRobot)
{
  EXPECT_THROW(
    SafetyCore(h12(), config(g1(), "sim_split"), false, kDomain), std::invalid_argument);
}
