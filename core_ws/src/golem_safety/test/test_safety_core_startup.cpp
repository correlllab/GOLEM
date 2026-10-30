#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
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

namespace
{

constexpr int kDomain = 91;
constexpr int kRealRobotDomain = 0;
constexpr double kStep = 0.01;  // s; the tests feed and step every 10 ms

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
  }
  return state;
}

// A controller's command: every body slot enabled at q with kp and kd.
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

bool contains(const std::string & text, const std::string & part)
{
  return text.find(part) != std::string::npos;
}

size_t count(const std::vector<Event> & events, Event::Kind kind)
{
  size_t n = 0;
  for (const auto & event : events) {
    n += event.kind == kind ? 1 : 0;
  }
  return n;
}

std::string texts(const std::vector<Event> & events)
{
  std::string out;
  for (const auto & event : events) {
    out += "[" + std::to_string(static_cast<int>(event.kind)) + "] " + event.text + "\n";
  }
  return out;
}

// Each JSON event whose "event" is `name`, parsed: (the event, its object).
std::vector<std::pair<Event, YAML::Node>> named(
  const std::vector<Event> & events, const std::string & name)
{
  std::vector<std::pair<Event, YAML::Node>> out;
  for (const auto & event : events) {
    if (event.text.empty() || event.text[0] != '{') {
      continue;
    }
    const YAML::Node node = YAML::Load(event.text);
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
  EXPECT_EQ(actual.motor_cmd, expected.motor_cmd);
  EXPECT_EQ(actual.reserve, expected.reserve);
  EXPECT_EQ(actual.crc, expected.crc);
}

// The inputs each mode reads, with their writers.
std::vector<std::pair<Input, const char *>> inputs_of(const SafetyConfig & cfg)
{
  switch (cfg.mode) {
    case golem_safety::Mode::FullBody:
      return {{Input::Full, "full"}};
    case golem_safety::Mode::Split:
      return {{Input::Lower, "lower"}, {Input::Upper, "upper"}};
  }
  return {};
}

// Feeds /lowstate and `cmd` on every input of the mode every 10 ms over steps first..last,
// stepping after each; returns the last output.
std::optional<LowCmd> run(
  SafetyCore & core, const RobotProfile & profile, const SafetyConfig & cfg, const LowCmd & cmd,
  int first, int last)
{
  std::optional<LowCmd> out;
  for (int i = first; i <= last; ++i) {
    const double t = i * kStep;
    core.on_state(state_of(profile), t);
    for (const auto & [input, writer] : inputs_of(cfg)) {
      core.on_command(input, cmd, writer, t);
    }
    out = core.step(t);
  }
  return out;
}

}  // namespace

TEST(Startup, ZeroStreamFromFirstStep)
{
  const LowCmd zero = golem_safety::zero_command(h12());
  for (const char * name : {"sim_safety_split", "default_safety_full"}) {
    for (const bool use_estop : {false, true}) {
      SCOPED_TRACE(std::string(name) + (use_estop ? ", use_estop" : ""));
      SafetyCore core(h12(), config(h12(), name), use_estop, kDomain);
      for (int i = 0; i < 100; ++i) {
        const auto out = core.step(i * kStep);
        ASSERT_TRUE(out) << "step " << i;
        expect_same(*out, zero);
      }
      EXPECT_FALSE(core.estopped());
    }
  }
}

TEST(Startup, NoLowstateNeverEngages)
{
  // H1-2: commands for 5 s and no /lowstate. zero_command on every step, and the reason
  // at most once a second.
  const LowCmd zero = golem_safety::zero_command(h12());
  SafetyCore core(h12(), config(h12(), "sim_safety_split"), false, kDomain);
  const LowCmd cmd = command(h12(), 0.1F);
  for (int i = 0; i < 500; ++i) {
    const double t = i * kStep;
    core.on_command(Input::Lower, cmd, "lower", t);
    core.on_command(Input::Upper, cmd, "upper", t);
    const auto out = core.step(t);
    ASSERT_TRUE(out) << "t = " << t;
    ASSERT_EQ(out->motor_cmd, zero.motor_cmd) << "t = " << t;
  }
  auto events = core.drain_events();
  auto refused = named(events, "engage_refused");
  ASSERT_EQ(refused.size(), 5U) << "at about 0, 1, 2, 3 and 4 s\n" << texts(events);
  for (const auto & [event, node] : refused) {
    EXPECT_EQ(event.kind, Event::Warn);
    EXPECT_EQ(
      event.text,
      "{\"event\": \"engage_refused\", \"mode\": \"split_mode\", "
      "\"reason\": \"no /lowstate received yet\"}");
  }

  // A /lowstate from the other robot: still zero_command, and why.
  LowState foreign = state_of(h12());
  foreign.mode_machine = 5;
  core.on_state(foreign, 5.5);
  core.on_command(Input::Upper, cmd, "upper", 5.5);
  auto out = core.step(5.5);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd, zero.motor_cmd);
  EXPECT_FALSE(core.estopped());
  refused = named(core.drain_events(), "engage_refused");
  ASSERT_EQ(refused.size(), 1U);
  EXPECT_EQ(
    refused[0].second["reason"].as<std::string>(), "/lowstate mode_machine is 5, expected 6");

  // The robot's own /lowstate: the next command engages its input. A command from before
  // does not, so the lower slots stay zero.
  core.on_state(state_of(h12()), 5.6);
  core.on_command(Input::Upper, cmd, "upper", 5.6);
  out = core.step(5.6);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd[20].q, 0.1F);
  EXPECT_EQ(out->motor_cmd[3], zero.motor_cmd[3]);
  EXPECT_FALSE(core.estopped());

  // G1: zero_command likewise.
  const LowCmd g1_zero = golem_safety::zero_command(g1());
  const SafetyConfig g1_cfg = config(g1(), "sim_split");
  SafetyCore g1_core(g1(), g1_cfg, false, kDomain);
  const LowCmd g1_cmd = command(g1(), 0.1F);
  for (int i = 0; i < 500; ++i) {
    const double t = i * kStep;
    g1_core.on_command(Input::Lower, g1_cmd, "lower", t);
    g1_core.on_command(Input::Upper, g1_cmd, "upper", t);
    const auto g1_out = g1_core.step(t);
    ASSERT_TRUE(g1_out) << "t = " << t;
    ASSERT_EQ(g1_out->motor_cmd, g1_zero.motor_cmd) << "t = " << t;
  }
  events = g1_core.drain_events();
  refused = named(events, "engage_refused");
  ASSERT_EQ(refused.size(), 5U) << texts(events);
  for (const auto & [event, node] : refused) {
    EXPECT_EQ(event.kind, Event::Warn);
    EXPECT_EQ(node["reason"].as<std::string>(), "no /lowstate received yet");
  }
  LowState h12_sample = state_of(g1());
  h12_sample.mode_machine = 6;
  g1_core.on_state(h12_sample, 5.5);
  g1_core.on_command(Input::Lower, g1_cmd, "lower", 5.5);
  g1_core.on_command(Input::Upper, g1_cmd, "upper", 5.5);
  auto g1_out = g1_core.step(5.5);
  ASSERT_TRUE(g1_out);
  EXPECT_EQ(g1_out->motor_cmd, g1_zero.motor_cmd);
  EXPECT_FALSE(g1_core.estopped());
  refused = named(g1_core.drain_events(), "engage_refused");
  ASSERT_EQ(refused.size(), 1U);
  EXPECT_EQ(
    refused[0].second["reason"].as<std::string>(), "/lowstate mode_machine is 6, expected 5");
  g1_core.on_state(state_of(g1()), 5.6);
  g1_core.on_command(Input::Lower, g1_cmd, "lower", 5.6);
  g1_core.on_command(Input::Upper, g1_cmd, "upper", 5.6);
  g1_out = g1_core.step(5.6);
  ASSERT_TRUE(g1_out);
  EXPECT_EQ(g1_out->motor_cmd[3].q, 0.1F);
  EXPECT_EQ(g1_out->motor_cmd[20].q, 0.1F);
  EXPECT_FALSE(g1_core.estopped());
}

TEST(Startup, ZeroStreamEngagesPerInput)
{
  const LowCmd zero = golem_safety::zero_command(h12());
  SafetyCore core(h12(), config(h12(), "sim_safety_split"), false, kDomain);
  core.on_state(state_of(h12()), 0.0);
  const LowCmd upper = command(h12(), 0.1F);
  core.on_command(Input::Upper, upper, "upper", 0.001);
  auto out = core.step(0.002);
  ASSERT_TRUE(out);
  for (int slot = 12; slot <= 26; ++slot) {
    EXPECT_EQ(out->motor_cmd[slot], upper.motor_cmd[slot]) << "slot " << slot;
  }
  for (int slot = 0; slot <= 11; ++slot) {
    EXPECT_EQ(out->motor_cmd[slot], zero.motor_cmd[slot]) << "slot " << slot;
  }
  auto engaged = named(core.drain_events(), "input_engaged");
  ASSERT_EQ(engaged.size(), 1U);
  EXPECT_EQ(engaged[0].first.kind, Event::Info);
  EXPECT_EQ(
    engaged[0].first.text,
    "{\"event\": \"input_engaged\", \"source\": \"split_mode_upper\", "
    "\"topic\": \"/safety/lowcmd_upper_in\"}");

  // Lower's first command engages Lower.
  const LowCmd lower = command(h12(), 0.05F);
  core.on_command(Input::Lower, lower, "lower", 0.003);
  out = core.step(0.004);
  ASSERT_TRUE(out);
  for (int slot = 0; slot <= 11; ++slot) {
    EXPECT_EQ(out->motor_cmd[slot], lower.motor_cmd[slot]) << "slot " << slot;
  }
  EXPECT_EQ(out->motor_cmd[20], upper.motor_cmd[20]);
  engaged = named(core.drain_events(), "input_engaged");
  ASSERT_EQ(engaged.size(), 1U);
  EXPECT_EQ(engaged[0].second["source"].as<std::string>(), "split_mode_lower");

  // A command that arrived before the first /lowstate does not engage its input then; the
  // input's next command does.
  SafetyCore early(h12(), config(h12(), "sim_safety_split"), false, kDomain);
  early.on_command(Input::Upper, upper, "upper", 0.0);
  early.on_state(state_of(h12()), 0.01);
  out = early.step(0.01);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd, zero.motor_cmd);
  early.on_command(Input::Upper, upper, "upper", 0.02);
  out = early.step(0.02);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd[20], upper.motor_cmd[20]);
}

TEST(Startup, G1StreamsZeroFromFirstStep)
{
  // No /lowstate and no input: zero_command on all 35 slots from the first step, stamped
  // with the G1's mode_machine.
  const LowCmd zero = golem_safety::zero_command(g1());
  SafetyCore core(g1(), config(g1(), "sim_split"), false, kDomain);
  for (int i = 0; i < 100; ++i) {
    const auto out = core.step(i * kStep);
    ASSERT_TRUE(out) << "step " << i;
    expect_same(*out, zero);
    ASSERT_EQ(out->mode_machine, 5);
    for (size_t slot = 0; slot < out->motor_cmd.size(); ++slot) {
      const auto & motor = out->motor_cmd[slot];
      ASSERT_EQ(motor.mode, 0) << "slot " << slot;
      ASSERT_TRUE(
        motor.q == 0.0F && motor.dq == 0.0F && motor.tau == 0.0F && motor.kp == 0.0F &&
        motor.kd == 0.0F) << "slot " << slot;
    }
  }
  EXPECT_FALSE(core.estopped());
  EXPECT_TRUE(core.drain_events().empty());
}

TEST(Startup, G1EngagesEachInputOnItsOwn)
{
  // The arm controller's commands alone, for 1 s: the waist and arms (12-28) carry them, and
  // the legs (0-11) stay at zero gain.
  const LowCmd zero = golem_safety::zero_command(g1());
  SafetyCore core(g1(), config(g1(), "sim_split"), false, kDomain);
  const LowCmd upper = command(g1(), 0.1F);
  for (int i = 0; i < 100; ++i) {
    const double t = i * kStep;
    core.on_state(state_of(g1()), t);
    core.on_command(Input::Upper, upper, "upper", t);
    const auto out = core.step(t);
    ASSERT_TRUE(out) << "t = " << t;
    for (int slot = 12; slot <= 28; ++slot) {
      ASSERT_EQ(out->motor_cmd[slot], upper.motor_cmd[slot]) << "t = " << t << " slot " << slot;
    }
    for (int slot = 0; slot <= 11; ++slot) {
      ASSERT_EQ(out->motor_cmd[slot], zero.motor_cmd[slot]) << "t = " << t << " slot " << slot;
    }
    ASSERT_EQ(out->mode_machine, 5);
  }
  auto events = core.drain_events();
  auto engaged = named(events, "input_engaged");
  ASSERT_EQ(engaged.size(), 1U) << texts(events);
  EXPECT_EQ(engaged[0].first.kind, Event::Info);
  EXPECT_EQ(
    engaged[0].first.text,
    "{\"event\": \"input_engaged\", \"source\": \"split_mode_upper\", "
    "\"topic\": \"/safety/lowcmd_upper_in\"}");
  EXPECT_EQ(count(events, Event::Warn), 0U) << texts(events);

  // The walking node's first valid command engages the legs.
  const LowCmd lower = command(g1(), 0.05F);
  core.on_state(state_of(g1()), 1.0);
  core.on_command(Input::Lower, lower, "lower", 1.0);
  core.on_command(Input::Upper, upper, "upper", 1.0);
  const auto out = core.step(1.0);
  ASSERT_TRUE(out);
  for (int slot = 0; slot <= 11; ++slot) {
    EXPECT_EQ(out->motor_cmd[slot], lower.motor_cmd[slot]) << "slot " << slot;
  }
  EXPECT_EQ(out->motor_cmd[20], upper.motor_cmd[20]);
  events = core.drain_events();
  engaged = named(events, "input_engaged");
  ASSERT_EQ(engaged.size(), 1U) << texts(events);
  EXPECT_EQ(engaged[0].second["source"].as<std::string>(), "split_mode_lower");
  EXPECT_FALSE(core.estopped());
}

TEST(Startup, G1LimitChecksFromBoot)
{
  // Before any input, a /lowstate with the left knee past limits.estop latches, in every G1
  // config. keep_writing then writes the estop command (every motor off) on every step,
  // and a valid command after the latch changes nothing.
  constexpr int kKnee = 3;
  for (const char * name : {"split", "sim_split", "full"}) {
    SCOPED_TRACE(name);
    const SafetyConfig cfg = config(g1(), name);
    SafetyCore core(g1(), cfg, false, kDomain);
    ASSERT_TRUE(core.step(0.0));
    LowState state = state_of(g1());
    state.motor_state[kKnee].q = static_cast<float>(cfg.limits.q_estop_low[kKnee] - 0.1);
    core.on_state(state, 0.002);
    ASSERT_TRUE(core.estopped());
    EXPECT_EQ(
      core.estop_reason().rfind("motor 3 (left_knee_joint) q out of estop range: ", 0), 0U) <<
      core.estop_reason();

    const LowCmd expected =
      golem_safety::estop_command(g1(), cfg, golem_safety::zero_command(g1()));
    for (int i = 0; i < 1000; ++i) {
      const double t = 0.004 + i * 0.002;
      for (const auto & [input, writer] : inputs_of(cfg)) {
        core.on_command(input, command(g1()), writer, t);
      }
      const auto out = core.step(t);
      ASSERT_TRUE(out) << "step " << i;
      expect_same(*out, expected);
    }
    const auto events = core.drain_events();
    const auto written = named(events, "estop_command_written");
    ASSERT_EQ(written.size(), 1U) << texts(events);
    EXPECT_EQ(written[0].first.kind, Event::Error);
    EXPECT_EQ(written[0].second["action_ms"].as<long long>(), 4);
    EXPECT_TRUE(named(events, "input_engaged").empty()) << texts(events);
  }
}

TEST(Estop, FirstSampleTriggeredLatches)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
  };
  for (const Case & c : {Case{h12(), "sim_safety_split"}, Case{g1(), "sim_split"}}) {
    SCOPED_TRACE(c.config_name);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    {
      SafetyCore core(c.profile, cfg, true, kDomain);
      core.on_estop(true, true, 0.0);
      ASSERT_TRUE(core.estopped());
      EXPECT_EQ(core.estop_reason(), "External estop triggered");
      // The estop command, the first of the H1-2's burst and of the G1's keep_writing.
      const auto out = core.step(0.002);
      ASSERT_TRUE(out);
      expect_same(*out, golem_safety::estop_command(c.profile, cfg, *out));
    }
    {
      SafetyCore core(c.profile, cfg, true, kDomain);
      core.on_estop(false, false, 0.0);
      ASSERT_TRUE(core.estopped());
      EXPECT_EQ(core.estop_reason(), "External estop unplugged");
    }
  }
}

TEST(Estop, NoEngageBeforeFirstSample)
{
  // H1-2: /lowstate and both inputs for 1 s, no estop sample: zero_command throughout.
  const LowCmd zero = golem_safety::zero_command(h12());
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  SafetyCore core(h12(), cfg, true, kDomain);
  const LowCmd cmd = command(h12(), 0.1F);
  for (int i = 0; i < 100; ++i) {
    const auto out = run(core, h12(), cfg, cmd, i, i);
    ASSERT_TRUE(out);
    ASSERT_EQ(out->motor_cmd, zero.motor_cmd) << "t = " << i * kStep;
  }
  const auto refused = named(core.drain_events(), "engage_refused");
  ASSERT_FALSE(refused.empty());
  EXPECT_EQ(
    refused[0].second["reason"].as<std::string>(),
    "no hardware estop sample received yet on /estop/triggered or /estop/plugged_in");

  // The first sample, neither triggered nor unplugged: the next commands engage.
  core.on_estop(false, true, 1.0);
  const auto out = run(core, h12(), cfg, cmd, 100, 100);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd[3].q, 0.1F);
  EXPECT_EQ(out->motor_cmd[20].q, 0.1F);
  EXPECT_FALSE(core.estopped());

  // G1: zero_command likewise, until the first sample.
  const LowCmd g1_zero = golem_safety::zero_command(g1());
  const SafetyConfig g1_cfg = config(g1(), "sim_split");
  SafetyCore g1_core(g1(), g1_cfg, true, kDomain);
  const LowCmd g1_cmd = command(g1(), 0.1F);
  for (int i = 0; i < 100; ++i) {
    const auto g1_out = run(g1_core, g1(), g1_cfg, g1_cmd, i, i);
    ASSERT_TRUE(g1_out);
    ASSERT_EQ(g1_out->motor_cmd, g1_zero.motor_cmd) << "t = " << i * kStep;
  }
  g1_core.on_estop(false, true, 1.0);
  const auto g1_out = run(g1_core, g1(), g1_cfg, g1_cmd, 100, 100);
  ASSERT_TRUE(g1_out);
  EXPECT_EQ(g1_out->motor_cmd[3].q, 0.1F);
  EXPECT_EQ(g1_out->motor_cmd[20].q, 0.1F);
}

TEST(Startup, RelayReadyNeedsTheRobotsLowstateAndAnEstopSample)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
  };
  for (const Case & c : {Case{h12(), "sim_safety_split"}, Case{g1(), "sim_split"}}) {
    SCOPED_TRACE(c.config_name);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    LowState foreign = state_of(c.profile);
    foreign.mode_machine = c.profile.mode_machine == 5 ? 6 : 5;

    // Without use_estop: the newest /lowstate decides, before any command.
    SafetyCore core(c.profile, cfg, false, kDomain);
    EXPECT_FALSE(core.relay_ready());
    core.on_state(foreign, 0.0);
    EXPECT_FALSE(core.relay_ready());
    core.on_state(state_of(c.profile), 0.01);
    EXPECT_TRUE(core.relay_ready());
    core.on_state(foreign, 0.02);
    EXPECT_FALSE(core.relay_ready()) << "a foreign /lowstate before engaging";
    core.on_state(state_of(c.profile), 0.03);
    EXPECT_TRUE(core.relay_ready());
    EXPECT_FALSE(core.estopped());

    // With use_estop, the first hardware estop sample too, in either order.
    SafetyCore estop(c.profile, cfg, true, kDomain);
    estop.on_state(state_of(c.profile), 0.0);
    EXPECT_FALSE(estop.relay_ready());
    estop.on_estop(false, true, 0.01);
    EXPECT_TRUE(estop.relay_ready());
    SafetyCore sample_first(c.profile, cfg, true, kDomain);
    sample_first.on_estop(false, true, 0.0);
    EXPECT_FALSE(sample_first.relay_ready());
    sample_first.on_state(state_of(c.profile), 0.01);
    EXPECT_TRUE(sample_first.relay_ready());

    // A first sample that latches never makes the relay ready.
    SafetyCore triggered(c.profile, cfg, true, kDomain);
    triggered.on_state(state_of(c.profile), 0.0);
    triggered.on_estop(true, true, 0.01);
    EXPECT_TRUE(triggered.estopped());
    EXPECT_FALSE(triggered.relay_ready());

    // The estop can latch before the relay is ready: a non-finite command, no /lowstate.
    SafetyCore early(c.profile, cfg, false, kDomain);
    LowCmd bad = command(c.profile);
    bad.motor_cmd[c.profile.upper_slots.first].q = std::numeric_limits<float>::quiet_NaN();
    early.on_command(Input::Upper, bad, "upper", 0.0);
    EXPECT_TRUE(early.estopped());
    EXPECT_FALSE(early.relay_ready());
    early.on_state(state_of(c.profile), 0.01);
    EXPECT_FALSE(early.relay_ready());
  }
}

TEST(Estop, StaleAfterEngageLatches)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
  };
  for (const Case & c : {Case{h12(), "sim_safety_split"}, Case{g1(), "sim_split"}}) {
    SCOPED_TRACE(c.config_name);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    ASSERT_EQ(cfg.estop.timeout_seconds, 0.5);
    SafetyCore core(c.profile, cfg, true, kDomain);
    // Engaged with a sample every 10 ms until t = 1.0; then the samples stop.
    for (int i = 0; i <= 100; ++i) {
      core.on_estop(false, true, i * kStep);
      ASSERT_TRUE(run(core, c.profile, cfg, command(c.profile), i, i));
    }
    ASSERT_FALSE(core.estopped());
    const double last_sample = 100 * kStep;
    for (int i = 101; i <= 149; ++i) {
      ASSERT_TRUE(run(core, c.profile, cfg, command(c.profile), i, i));
      ASSERT_FALSE(core.estopped()) << "t = " << i * kStep;
    }
    for (int i = 150; i <= 160 && !core.estopped(); ++i) {
      run(core, c.profile, cfg, command(c.profile), i, i);
    }
    ASSERT_TRUE(core.estopped()) << "0.6 s after the last sample";
    const std::string & reason = core.estop_reason();
    EXPECT_EQ(reason.rfind("External estop silent: no sample on /estop/triggered or ", 0), 0U) <<
      reason;
    EXPECT_TRUE(contains(reason, "(timeout 0.5 s)")) << reason;
    // A sample after the latch changes nothing.
    core.on_estop(false, true, last_sample + 0.7);
    EXPECT_TRUE(core.estopped());
  }

  // From the first sample on, also before engaging.
  SafetyCore early(h12(), config(h12(), "sim_safety_split"), true, kDomain);
  early.on_estop(false, true, 0.0);
  ASSERT_TRUE(early.step(0.5));
  EXPECT_FALSE(early.estopped());
  ASSERT_TRUE(early.step(0.6));
  EXPECT_TRUE(early.estopped());
}

TEST(Estop, IgnoredWithoutUseEstop)
{
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  SafetyCore core(h12(), cfg, false, kDomain);
  core.on_estop(true, true, 0.0);
  core.on_estop(false, false, 0.0);
  EXPECT_FALSE(core.estopped());
  // Engages with no sample, and no sample for 2 s latches nothing.
  const auto out = run(core, h12(), cfg, command(h12(), 0.1F), 1, 200);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->motor_cmd[20].q, 0.1F);
  EXPECT_FALSE(core.estopped()) << core.estop_reason();

  const SafetyConfig g1_cfg = config(g1(), "sim_split");
  SafetyCore g1_core(g1(), g1_cfg, false, kDomain);
  g1_core.on_estop(true, false, 0.0);
  EXPECT_TRUE(run(g1_core, g1(), g1_cfg, command(g1()), 1, 100));
  EXPECT_FALSE(g1_core.estopped());
}

TEST(Quiet, InputHeldAndReported)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
    Input quiet;
    const char * source;
    const char * topic;
    double seconds;  // the input's stale time
    int last_before, first_after;  // the gap, in 10 ms steps
  };
  const std::vector<Case> cases = {
    {h12(), "sim_safety_split", Input::Lower, "split_mode_lower", "/safety/lowcmd_lower_in",
      0.2, 0, 101},
    // A gap of 3 s, past stale.upper_seconds (2.0).
    {h12(), "default_safety_split", Input::Upper, "split_mode_upper", "/safety/lowcmd_upper_in",
      2.0, 0, 301},
    {g1(), "split", Input::Lower, "split_mode_lower", "/safety/lowcmd_lower_in", 0.2, 0, 101},
    {g1(), "split", Input::Upper, "split_mode_upper", "/safety/lowcmd_upper_in", 2.0, 0, 301},
  };
  for (const Case & c : cases) {
    SCOPED_TRACE(std::string(c.config_name) + " " + c.topic);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    SafetyCore core(c.profile, cfg, false, kDomain);
    const LowCmd held = command(c.profile, 0.05F);
    const LowCmd live = command(c.profile, 0.1F);
    const golem_safety::SlotRange & slots =
      c.quiet == Input::Lower ? c.profile.lower_slots : c.profile.upper_slots;
    const Input other = c.quiet == Input::Lower ? Input::Upper : Input::Lower;
    for (int i = 0; i <= c.first_after + 10; ++i) {
      const double t = i * kStep;
      core.on_state(state_of(c.profile), t);
      core.on_command(other, live, other == Input::Lower ? "lower" : "upper", t);
      if (i <= c.last_before || i >= c.first_after) {
        core.on_command(c.quiet, held, c.quiet == Input::Lower ? "lower" : "upper", t);
      }
      const auto out = core.step(t);
      ASSERT_TRUE(out) << "t = " << t;
      for (int slot = slots.first; slot <= slots.last; ++slot) {
        ASSERT_EQ(out->motor_cmd[slot], held.motor_cmd[slot]) << "t = " << t << " slot " << slot;
      }
      ASSERT_FALSE(core.estopped()) << core.estop_reason();
      if (i == 0) {
        core.drain_events();  // engaging
      }
    }
    const auto events = core.drain_events();
    EXPECT_EQ(count(events, Event::Warn), 1U) << texts(events);
    const auto quiet = named(events, "input_quiet");
    ASSERT_EQ(quiet.size(), 1U) << texts(events);
    EXPECT_EQ(quiet[0].first.kind, Event::Warn);
    const YAML::Node & q = quiet[0].second;
    EXPECT_EQ(
      keys_of(q), (std::vector<std::string>{"age", "event", "seconds", "source", "topic"}));
    EXPECT_GT(q["age"].as<double>(), c.seconds);
    EXPECT_LE(q["age"].as<double>(), c.seconds + kStep + 1e-9);
    EXPECT_EQ(q["seconds"].as<double>(), c.seconds);
    EXPECT_EQ(q["source"].as<std::string>(), c.source);
    EXPECT_EQ(q["topic"].as<std::string>(), c.topic);
    const auto resumed = named(events, "input_resumed");
    ASSERT_EQ(resumed.size(), 1U) << texts(events);
    EXPECT_EQ(resumed[0].first.kind, Event::Info);
    const YAML::Node & r = resumed[0].second;
    EXPECT_EQ(keys_of(r), (std::vector<std::string>{"event", "gap", "source", "topic"}));
    EXPECT_NEAR(r["gap"].as<double>(), (c.first_after - c.last_before) * kStep, 1e-4);
    EXPECT_EQ(r["source"].as<std::string>(), c.source);
  }
}

TEST(Quiet, FullInputUsesLowerSeconds)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
  };
  for (const Case & c : {Case{h12(), "default_safety_full"}, Case{g1(), "full"}}) {
    SCOPED_TRACE(c.config_name);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    ASSERT_EQ(cfg.stale.lower_seconds, 0.2);
    ASSERT_LE(cfg.stale.upper_seconds, 0.0);
    SafetyCore core(c.profile, cfg, false, kDomain);
    const LowCmd held = command(c.profile, 0.05F);
    // The full input sends at t = 0, then from t = 1.0.
    for (int i = 0; i <= 110; ++i) {
      const double t = i * kStep;
      core.on_state(state_of(c.profile), t);
      if (i == 0 || i >= 100) {
        core.on_command(Input::Full, held, "full", t);
      }
      const auto out = core.step(t);
      ASSERT_TRUE(out) << "t = " << t;
      ASSERT_EQ(out->motor_cmd[3], held.motor_cmd[3]) << "t = " << t;
      ASSERT_FALSE(core.estopped()) << core.estop_reason();
      if (i == 0) {
        core.drain_events();
      }
    }
    const auto events = core.drain_events();
    EXPECT_EQ(count(events, Event::Warn), 1U) << texts(events);
    const auto quiet = named(events, "input_quiet");
    ASSERT_EQ(quiet.size(), 1U) << texts(events);
    const YAML::Node & q = quiet[0].second;
    EXPECT_EQ(q["seconds"].as<double>(), 0.2);
    EXPECT_GT(q["age"].as<double>(), 0.2);
    EXPECT_LE(q["age"].as<double>(), 0.2 + kStep + 1e-9);
    EXPECT_EQ(q["source"].as<std::string>(), "full_body_mode");
    EXPECT_EQ(q["topic"].as<std::string>(), "/safety/lowcmd_in");
    EXPECT_EQ(named(events, "input_resumed").size(), 1U) << texts(events);
  }
}

TEST(Quiet, LowstateGapHolds)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
  };
  for (const Case & c : {Case{h12(), "sim_safety_split"}, Case{g1(), "split"}}) {
    SCOPED_TRACE(c.config_name);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    ASSERT_EQ(cfg.stale.state_seconds, 0.1);
    SafetyCore core(c.profile, cfg, false, kDomain);
    const LowCmd cmd = command(c.profile, 0.05F);
    // /lowstate stops for 1 s, from t = 0.5 to t = 1.5.
    for (int i = 0; i <= 200; ++i) {
      const double t = i * kStep;
      if (i <= 50 || i >= 150) {
        core.on_state(state_of(c.profile), t);
      }
      core.on_command(Input::Lower, cmd, "lower", t);
      core.on_command(Input::Upper, cmd, "upper", t);
      const auto out = core.step(t);
      ASSERT_TRUE(out) << "t = " << t;
      ASSERT_EQ(out->motor_cmd[3].q, 0.05F) << "t = " << t;
      ASSERT_EQ(out->motor_cmd[20].q, 0.05F) << "t = " << t;
      ASSERT_FALSE(core.estopped()) << core.estop_reason();
      if (i == 0) {
        core.drain_events();  // engaging
      }
    }
    const auto events = core.drain_events();
    EXPECT_EQ(count(events, Event::Warn), 1U) << texts(events);
    const auto quiet = named(events, "state_quiet");
    ASSERT_EQ(quiet.size(), 1U) << texts(events);
    EXPECT_EQ(quiet[0].first.kind, Event::Warn);
    EXPECT_EQ(
      keys_of(quiet[0].second), (std::vector<std::string>{"age", "event", "seconds", "topic"}));
    EXPECT_EQ(quiet[0].second["topic"].as<std::string>(), "/lowstate");
    EXPECT_EQ(quiet[0].second["seconds"].as<double>(), 0.1);
    const auto resumed = named(events, "state_resumed");
    ASSERT_EQ(resumed.size(), 1U) << texts(events);
    EXPECT_EQ(resumed[0].first.kind, Event::Info);
    EXPECT_EQ(
      keys_of(resumed[0].second), (std::vector<std::string>{"event", "gap", "topic"}));
    EXPECT_NEAR(resumed[0].second["gap"].as<double>(), 1.0, 1e-4);
  }
}

TEST(Lag, Domain0LateTicksLatch)
{
  // /lowstate every 2 ms on time for 1 s, then each sample 1 ms later against the robot's
  // ms tick than the one before. The baseline may rise 0.5 ms/s, so the lag at the k-th
  // late sample is 0.9985 k ms: over 100 ms first at k = 101.
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  ASSERT_EQ(cfg.control.state_lag_ms, 100.0);
  auto feed = [](SafetyCore & core, LowState state, int late_samples) {
      for (int i = 0; i <= 500; ++i) {
        state.tick = static_cast<uint32_t>(2 * i);
        core.on_state(state, 0.002 * i);
      }
      for (int k = 1; k <= late_samples && !core.estopped(); ++k) {
        state.tick = static_cast<uint32_t>(1000 + 2 * k);
        core.on_state(state, 1.0 + 0.003 * k);
      }
    };
  {
    SafetyCore core(h12(), cfg, false, kRealRobotDomain);
    feed(core, state_of(h12()), 100);
    EXPECT_FALSE(core.estopped()) << core.estop_reason();
    LowState late = state_of(h12());
    late.tick = 1000 + 2 * 101;
    core.on_state(late, 1.0 + 0.003 * 101);
    ASSERT_TRUE(core.estopped());
    EXPECT_EQ(
      core.estop_reason(), "/lowstate arriving 101 ms late (> 100 ms) against the robot tick");
  }
  {
    // Off the real robot's domain the sims' ticks are not checked.
    SafetyCore core(h12(), cfg, false, kDomain);
    feed(core, state_of(h12()), 1000);
    EXPECT_FALSE(core.estopped()) << core.estop_reason();
  }
  {
    // The G1 configs carry state_lag_ms 0: its check stays off, engaged on domain 0 too.
    const SafetyConfig g1_cfg = config(g1(), "split");
    ASSERT_EQ(g1_cfg.control.state_lag_ms, 0.0);
    SafetyCore core(g1(), g1_cfg, false, kRealRobotDomain);
    ASSERT_TRUE(run(core, g1(), g1_cfg, command(g1()), 0, 0)) << "engaged";
    feed(core, state_of(g1()), 1000);
    EXPECT_FALSE(core.estopped()) << core.estop_reason();
  }
}

TEST(Lag, TickWrapAndSlowDriftAreNotLag)
{
  const SafetyConfig cfg = config(h12(), "sim_safety_split");
  // The uint32 ms tick wraps 1 s in.
  {
    SafetyCore core(h12(), cfg, false, kRealRobotDomain);
    LowState state = state_of(h12());
    const uint32_t start = 0xFFFFFFFFU - 999U;
    for (int i = 0; i <= 1000; ++i) {
      state.tick = start + static_cast<uint32_t>(2 * i);
      core.on_state(state, 0.002 * i);
    }
    EXPECT_FALSE(core.estopped()) << core.estop_reason();
  }
  // The host clock gains 0.4 ms/s on the tick for 10 min (240 ms in all): the baseline
  // follows drift of up to 0.5 ms/s.
  {
    SafetyCore core(h12(), cfg, false, kRealRobotDomain);
    LowState state = state_of(h12());
    for (int i = 0; i <= 6000; ++i) {
      state.tick = static_cast<uint32_t>(100 * i);
      core.on_state(state, 0.10004 * i);
    }
    EXPECT_FALSE(core.estopped()) << core.estop_reason();
  }
}

TEST(Exit, SplitStopsAtOnce)
{
  struct Case
  {
    const RobotProfile & profile;
    const char * config_name;
  };
  for (const Case & c : {Case{h12(), "sim_safety_split"}, Case{h12(), "default_safety_full"},
      Case{g1(), "sim_split"}, Case{g1(), "full"}})
  {
    SCOPED_TRACE(c.config_name);
    const SafetyConfig cfg = config(c.profile, c.config_name);
    SafetyCore core(c.profile, cfg, false, kDomain);
    const auto out = run(core, c.profile, cfg, command(c.profile, 0.05F), 0, 10);
    ASSERT_TRUE(out);
    ASSERT_EQ(out->motor_cmd[3].q, 0.05F);
    // The robot holds the last command written; nothing more is published.
    EXPECT_FALSE(core.exit_step(0.11));
    EXPECT_FALSE(core.exit_step(0.12));
    EXPECT_FALSE(core.step(0.13));
    EXPECT_FALSE(core.estopped());
  }
}
