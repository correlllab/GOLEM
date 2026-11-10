#include <gtest/gtest.h>

#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"

using golem_safety::EstopAfter;
using golem_safety::EstopMode;
using golem_safety::Mode;
using golem_safety::RobotProfile;
using golem_safety::SafetyConfig;

namespace {

// Return the installed H1-2 profile.
const RobotProfile & h12()
{
  static const RobotProfile profile = golem_safety::load_robot_profile("h12");
  return profile;
}

// Return the installed G1 profile.
const RobotProfile & g1()
{
  static const RobotProfile profile = golem_safety::load_robot_profile("g1");
  return profile;
}

// Every config under config/h12/.
const char * const kH12Configs[] = {
  "default_safety_full", "default_safety_split", "locomotion_safety_full", "relax_safety_full",
  "relax_safety_split", "sim_safety_split", "tight_safety_full", "tight_safety_split"};

// Every config under config/g1/.
const char * const kG1Configs[] = {"full", "sim_split", "split"};

// Return the message of the std::runtime_error `fn` throws; fail the test when it throws
// nothing.
std::string error_of(const std::function<void()> & fn)
{
  try {
    fn();
  } catch (const std::runtime_error & e) {
    return e.what();
  }
  ADD_FAILURE() << "expected std::runtime_error";
  return "";
}

// Return whether `part` occurs in `text`.
bool contains(const std::string & text, const std::string & part)
{
  return text.find(part) != std::string::npos;
}

// Write a config file in a private temporary directory, removed with the object.
class TempConfig
{
public:
  explicit TempConfig(const std::string & text)
  {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
      ("golem_safety_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    path_ = (dir_ / "case.yaml").string();
    std::ofstream(path_) << text;
  }
  ~TempConfig() {std::filesystem::remove_all(dir_);}
  TempConfig(const TempConfig &) = delete;
  TempConfig & operator=(const TempConfig &) = delete;
  const std::string & path() const {return path_;}

private:
  std::filesystem::path dir_;
  std::string path_;
};

// Return `text` plus whichever of control.state_lag_ms and estop_response.after it leaves
// out, with the robot's values: those keys have no default, and most tests are not about them.
std::string completed(const std::string & text, const RobotProfile & profile)
{
  const bool g1 = profile.robot == "g1";
  std::string out = text;
  if (!contains(text, "control:")) {
    out += g1 ? "control: {state_lag_ms: 0.0}\n" : "control: {state_lag_ms: 100.0}\n";
  }
  if (!contains(text, "estop_response:")) {
    out += g1 ? "estop_response: {after: keep_writing}\n" :
      "estop_response: {after: burst_then_silence}\n";
  }
  return out;
}

// Parse `text` as it is.
SafetyConfig parse_raw(const std::string & text, const RobotProfile & profile)
{
  return golem_safety::parse_safety_config(text, "case.yaml", profile);
}

// Return the error parsing `text` as it is throws.
std::string parse_raw_error(const std::string & text, const RobotProfile & profile)
{
  return error_of([&] {parse_raw(text, profile);});
}

// Parse completed(`text`).
SafetyConfig parse(const std::string & text, const RobotProfile & profile)
{
  return parse_raw(completed(text, profile), profile);
}

// Return the error parsing completed(`text`) throws.
std::string parse_error(const std::string & text, const RobotProfile & profile)
{
  return error_of([&] {parse(text, profile);});
}

// Return a YAML list of `count` copies of `value`.
std::string list_of(double value, size_t count)
{
  std::string text = "[";
  for (size_t i = 0; i < count; ++i) {
    text += (i ? ", " : "") + std::to_string(value);
  }
  return text + "]";
}

}  // namespace

// A bare name, with or without .yaml, resolves under config/<robot>/; a path is used as
// given; a missing name reports the robot and the file looked for.
TEST(Config, ResolvesBareNameUnderRobotDir)
{
  const SafetyConfig plain = golem_safety::load_safety_config("default_safety_split", h12());
  const SafetyConfig suffixed =
    golem_safety::load_safety_config("default_safety_split.yaml", h12());
  EXPECT_EQ(plain.robot, "h12");
  EXPECT_EQ(plain.mode, Mode::Split);
  EXPECT_EQ(plain.source, suffixed.source);
  EXPECT_TRUE(contains(plain.source, "/config/h12/default_safety_split.yaml")) << plain.source;
  EXPECT_EQ(plain.limits.q_clip_low, suffixed.limits.q_clip_low);

  const SafetyConfig split = golem_safety::load_safety_config("split", g1());
  EXPECT_EQ(split.robot, "g1");
  EXPECT_EQ(split.mode, Mode::Split);
  EXPECT_TRUE(contains(split.source, "/config/g1/split.yaml")) << split.source;

  const TempConfig file(completed("mode: split_mode\n", h12()));
  EXPECT_EQ(golem_safety::load_safety_config(file.path(), h12()).source, file.path());

  const std::string message =
    error_of([] {golem_safety::load_safety_config("split", h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, "/config/h12/split.yaml")) << message;
}

// An unknown key, at the root or nested, is refused naming the robot, the file and the key.
TEST(Config, UnknownKeyThrows)
{
  const TempConfig top(completed("mode: split_mode\nbogus: 1\n", h12()));
  std::string message = error_of([&] {golem_safety::load_safety_config(top.path(), h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, top.path())) << message;
  EXPECT_TRUE(contains(message, "'bogus'")) << message;

  const TempConfig nested(
    completed("mode: split_mode\nlimits:\n  clip:\n    kp_maxx: 1.0\n", h12()));
  message = error_of([&] {golem_safety::load_safety_config(nested.path(), h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, nested.path())) << message;
  EXPECT_TRUE(contains(message, "'limits.clip.kp_maxx'")) << message;

  // limits.estop takes no gain limits.
  EXPECT_TRUE(
    contains(parse_error("limits:\n  estop:\n    kp_max: 1.0\n", h12()), "'limits.estop.kp_max'"));
}

// A mode the profile lacks is refused, listing the profile's modes.
TEST(Config, ModeNotInProfileThrows)
{
  const TempConfig file("mode: legs_only_mode\n");
  const std::string message =
    error_of([&] {golem_safety::load_safety_config(file.path(), h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, file.path())) << message;
  EXPECT_TRUE(contains(message, "'mode'")) << message;
  EXPECT_TRUE(
    contains(message, "'legs_only_mode' is not a mode of robot h12 (full_body_mode, split_mode)"))
    << message;

  EXPECT_TRUE(contains(parse_error("mode: bogus_mode\n", h12()), "'mode'"));

  // A mode name the loader knows is refused the same way by a profile without it.
  RobotProfile split_only = g1();
  split_only.modes = {Mode::Split};
  EXPECT_TRUE(
    contains(
      parse_error("mode: full_body_mode\n", split_only),
      "'full_body_mode' is not a mode of robot g1 (split_mode)"));

  for (const RobotProfile * profile : {&h12(), &g1()}) {
    EXPECT_EQ(parse("mode: split_mode\n", *profile).mode, Mode::Split);
    EXPECT_EQ(parse("mode: full_body_mode\n", *profile).mode, Mode::FullBody);
  }
}

// A config without a mode runs the profile's first mode.
TEST(Config, ModeDefaultsToTheProfilesFirstMode)
{
  EXPECT_EQ(parse("", h12()).mode, Mode::FullBody);
  EXPECT_EQ(parse("", g1()).mode, Mode::Split);
}

// A per-joint list must have one entry per body joint of the robot, and applies joint by
// joint.
TEST(Config, ListLengthMustMatchJoints)
{
  const TempConfig short_list(
    completed("mode: split_mode\nlimits:\n  clip:\n    velocity_ratio: [0.1, 0.1]\n", h12()));
  std::string message =
    error_of([&] {golem_safety::load_safety_config(short_list.path(), h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, short_list.path())) << message;
  EXPECT_TRUE(contains(message, "'limits.clip.velocity_ratio'")) << message;
  EXPECT_TRUE(contains(message, "27")) << message;

  message = parse_error("limits:\n  estop:\n    torque_ratio: " + list_of(0.5, 29) + "\n", h12());
  EXPECT_TRUE(contains(message, "'limits.estop.torque_ratio'")) << message;
  message = parse_error("limits:\n  clip:\n    position_offset: " + list_of(0.0, 27) + "\n", g1());
  EXPECT_TRUE(contains(message, "'limits.clip.position_offset'")) << message;
  EXPECT_TRUE(contains(message, "29")) << message;

  const SafetyConfig ok =
    parse("limits:\n  clip:\n    velocity_ratio: " + list_of(0.25, 27) + "\n", h12());
  ASSERT_EQ(ok.limits.dq_clip.size(), 27u);
  EXPECT_DOUBLE_EQ(ok.limits.dq_clip[0], h12().joints[0].dq_max * 0.25);
  EXPECT_DOUBLE_EQ(ok.limits.dq_clip[26], h12().joints[26].dq_max * 0.25);

  // The damp gains are per joint too.
  message = parse_error(
    "estop_response: {mode: damp, after: keep_writing, kd: " + list_of(1.0, 3) + "}\n", g1());
  EXPECT_TRUE(contains(message, "'estop_response.kd'")) << message;
}

// SAFETY: every H1-2 config writes a burst of 50 zero commands after an estop, with the
// H1-2's stale times, lag check and topics.
TEST(Config, CarriedH12ConfigsAreZeroStreamAndBurst)
{
  for (const char * name : kH12Configs) {
    const SafetyConfig config = golem_safety::load_safety_config(name, h12());
    EXPECT_EQ(config.estop_response.mode, EstopMode::Zero) << name;
    EXPECT_EQ(config.estop_response.after, EstopAfter::BurstThenSilence) << name;
    EXPECT_EQ(config.estop_response.burst_count, 50) << name;
    EXPECT_DOUBLE_EQ(config.stale.lower_seconds, 0.2) << name;
    EXPECT_DOUBLE_EQ(config.stale.state_seconds, 0.1) << name;
    EXPECT_DOUBLE_EQ(config.control.state_lag_ms, 100.0) << name;
    EXPECT_DOUBLE_EQ(config.control.publish_hz, 500.0) << name;
    EXPECT_FALSE(config.logging.enabled) << name;
    EXPECT_EQ(config.logging.base_dir, "/tmp/golem_safety") << name;
    EXPECT_EQ(config.logging.chunk_prefix, "chunk_") << name;
    EXPECT_EQ(config.topics.low_cmd_out, "/lowcmd") << name;
    EXPECT_EQ(config.topics.low_state, "/lowstate") << name;
  }
}

// control.state_lag_ms has no default: leaving it out is refused, naming robot and file.
TEST(Config, StateLagMsIsRequired)
{
  const std::string h12_rest = "estop_response: {after: burst_then_silence}\n";
  const std::string g1_rest = "estop_response: {after: keep_writing}\n";
  std::string message = parse_raw_error(h12_rest, h12());
  EXPECT_TRUE(contains(message, "'control.state_lag_ms'")) << message;
  EXPECT_TRUE(contains(message, "must be given")) << message;
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, "case.yaml")) << message;
  message = parse_raw_error("control: {publish_hz: 500.0}\n" + g1_rest, g1());
  EXPECT_TRUE(contains(message, "'control.state_lag_ms'")) << message;
  EXPECT_TRUE(contains(message, "g1")) << message;

  EXPECT_DOUBLE_EQ(
    parse_raw("control: {state_lag_ms: 100.0}\n" + h12_rest, h12()).control.state_lag_ms, 100.0);
  EXPECT_DOUBLE_EQ(
    parse_raw("control: {state_lag_ms: 0}\n" + g1_rest, g1()).control.state_lag_ms, 0.0);
  EXPECT_TRUE(
    contains(
      parse_raw_error("control: {state_lag_ms: .nan}\n" + h12_rest, h12()), "state_lag_ms"));

  const TempConfig file("mode: split_mode\n" + h12_rest);
  message = error_of([&] {golem_safety::load_safety_config(file.path(), h12());});
  EXPECT_TRUE(contains(message, file.path())) << message;
  EXPECT_TRUE(contains(message, "'control.state_lag_ms'")) << message;
}

// estop_response.after has no default; burst_count keeps its default of 50.
TEST(Config, EstopResponseAfterIsRequired)
{
  const std::string lag = "control: {state_lag_ms: 100.0}\n";
  std::string message = parse_raw_error(lag, h12());
  EXPECT_TRUE(contains(message, "'estop_response.after'")) << message;
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, "burst_then_silence")) << message;

  EXPECT_TRUE(
    contains(
      parse_raw_error(lag + "estop_response: {mode: zero}\n", h12()), "'estop_response.after'"));
  EXPECT_TRUE(contains(parse_raw_error(lag, g1()), "'estop_response.after'"));
  EXPECT_EQ(
    parse("estop_response: {after: keep_writing}\n", h12()).estop_response.after,
    EstopAfter::KeepWriting);
  EXPECT_EQ(
    parse("estop_response: {after: burst_then_silence}\n", h12()).estop_response.after,
    EstopAfter::BurstThenSilence);
  EXPECT_EQ(parse("", h12()).estop_response.burst_count, 50);
}

// The H1-2 split configs' upper_seconds are 2.0 (default, tight) and 0.0 (relax, sim).
TEST(Config, H12UpperStaleSecondsAreTheOldEffectiveValues)
{
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("default_safety_split", h12())
    .stale.upper_seconds, 2.0);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("tight_safety_split", h12())
    .stale.upper_seconds, 2.0);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("relax_safety_split", h12())
    .stale.upper_seconds, 0.0);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("sim_safety_split", h12())
    .stale.upper_seconds, 0.0);
}

// SAFETY: every G1 config writes the zero estop command until restart, logs, and runs
// without the lag check.
TEST(Config, G1ConfigsKeepWriting)
{
  for (const char * name : kG1Configs) {
    const SafetyConfig config = golem_safety::load_safety_config(name, g1());
    EXPECT_EQ(config.estop_response.after, EstopAfter::KeepWriting) << name;
    EXPECT_EQ(config.estop_response.mode, EstopMode::Zero) << name;
    EXPECT_TRUE(config.logging.enabled) << name;
    EXPECT_EQ(config.logging.base_dir, "/tmp/golem_safety") << name;
    EXPECT_DOUBLE_EQ(config.control.state_lag_ms, 0.0) << name;
    EXPECT_DOUBLE_EQ(config.limits.kp_max, 200.0) << name;
    EXPECT_DOUBLE_EQ(config.limits.kd_max, 10.0) << name;
  }
  const SafetyConfig split = golem_safety::load_safety_config("split", g1());
  EXPECT_EQ(split.mode, Mode::Split);
  EXPECT_EQ(split.topics.low_cmd_lower_in, "/safety/lowcmd_lower_in");
  EXPECT_EQ(split.topics.low_cmd_upper_in, "/safety/lowcmd_upper_in");
  EXPECT_EQ(split.topics.low_cmd_out, "/lowcmd");
  ASSERT_EQ(split.estop_response.kd.size(), 29u);
  EXPECT_DOUBLE_EQ(split.estop_response.kd[7], 8.0);
  EXPECT_DOUBLE_EQ(split.stale.upper_seconds, 2.0);
  EXPECT_DOUBLE_EQ(split.stale.lower_seconds, 0.2);

  const SafetyConfig sim = golem_safety::load_safety_config("sim_split", g1());
  EXPECT_EQ(sim.mode, Mode::Split);
  EXPECT_EQ(sim.topics.low_cmd_out, "/lowcmd");
  EXPECT_DOUBLE_EQ(sim.stale.lower_seconds, 1.0);

  const SafetyConfig full = golem_safety::load_safety_config("full", g1());
  EXPECT_EQ(full.mode, Mode::FullBody);
  EXPECT_EQ(full.topics.low_cmd_out, "/lowcmd");
}

// Every installed config loads for its robot.
TEST(Config, EveryCarriedConfigLoads)
{
  for (const char * name : kH12Configs) {
    EXPECT_NO_THROW(golem_safety::load_safety_config(name, h12())) << name;
  }
  for (const char * name : kG1Configs) {
    EXPECT_NO_THROW(golem_safety::load_safety_config(name, g1())) << name;
  }
}

// A config name resolves only for its own robot.
TEST(Config, ConfigNamesBelongToTheirRobot)
{
  for (const char * name : kH12Configs) {
    EXPECT_THROW(golem_safety::load_safety_config(name, g1()), std::runtime_error) << name;
  }
  for (const char * name : kG1Configs) {
    EXPECT_THROW(golem_safety::load_safety_config(name, h12()), std::runtime_error) << name;
  }
}

// SAFETY: a path under another robot's config directory is refused, since several configs
// of one robot also parse for the other.
TEST(Config, PathInAnotherRobotsDirectoryIsRefused)
{
  // The G1's full.yaml parses for the H1-2, and the H1-2's sim_safety_split.yaml for the G1.
  const std::string share = ament_index_cpp::get_package_share_directory("golem_safety");
  for (const char * name : kG1Configs) {
    const std::string path = share + "/config/g1/" + name + ".yaml";
    const std::string message =
      error_of([&] {golem_safety::load_safety_config(path, h12());});
    EXPECT_TRUE(contains(message, "h12")) << message;
    EXPECT_TRUE(contains(message, path)) << message;
    EXPECT_TRUE(contains(message, "robot g1")) << message;
  }
  for (const char * name : kH12Configs) {
    const std::string path = share + "/config/h12/" + name + ".yaml";
    const std::string message =
      error_of([&] {golem_safety::load_safety_config(path, g1());});
    EXPECT_TRUE(contains(message, "robot h12")) << message;
  }
  EXPECT_NO_THROW(
    golem_safety::load_safety_config(share + "/config/h12/sim_safety_split.yaml", h12()));
  const TempConfig file(completed("mode: split_mode\n", h12()));
  EXPECT_NO_THROW(golem_safety::load_safety_config(file.path(), h12()));
}

// A config without limits derives them from the defaults: offset 0.01 rad, ratios 0.5,
// kp_max 300, kd_max 30.
TEST(Config, DerivedLimitsUseTheSchemaDefaults)
{
  const SafetyConfig c = parse("mode: split_mode\n", h12());
  ASSERT_EQ(c.limits.q_clip_low.size(), 27u);
  for (size_t i = 0; i < 27; ++i) {
    const auto & joint = h12().joints[i];
    EXPECT_DOUBLE_EQ(c.limits.q_clip_low[i], joint.q_low + 0.01) << i;
    EXPECT_DOUBLE_EQ(c.limits.q_clip_high[i], joint.q_high - 0.01) << i;
    EXPECT_DOUBLE_EQ(c.limits.dq_clip[i], joint.dq_max * 0.5) << i;
    EXPECT_DOUBLE_EQ(c.limits.tau_clip[i], joint.tau_max * 0.5) << i;
    EXPECT_DOUBLE_EQ(c.limits.q_estop_low[i], joint.q_low + 0.01) << i;
    EXPECT_DOUBLE_EQ(c.limits.q_estop_high[i], joint.q_high - 0.01) << i;
    EXPECT_DOUBLE_EQ(c.limits.dq_estop[i], joint.dq_max * 0.5) << i;
    EXPECT_DOUBLE_EQ(c.limits.tau_estop[i], joint.tau_max * 0.5) << i;
  }
  EXPECT_DOUBLE_EQ(c.limits.kp_max, 300.0);
  EXPECT_DOUBLE_EQ(c.limits.kd_max, 30.0);
  EXPECT_EQ(parse("", g1()).limits.q_clip_low.size(), 29u);
}

// A negative offset widens a range; an offset that empties a joint's range is refused.
TEST(Config, NegativePositionOffsetWidensAndTooLargeThrows)
{
  const SafetyConfig wide =
    parse("limits:\n  estop: {position_offset: -1.01}\n", h12());
  EXPECT_DOUBLE_EQ(wide.limits.q_estop_low[0], h12().joints[0].q_low - 1.01);
  EXPECT_DOUBLE_EQ(wide.limits.q_estop_high[0], h12().joints[0].q_high + 1.01);

  // left_ankle_roll_joint spans +-0.261799, so an offset of 0.3 inverts it.
  const std::string message =
    parse_error("limits:\n  clip: {position_offset: 0.3}\n", h12());
  EXPECT_TRUE(contains(message, "'limits.clip.position_offset'")) << message;
  EXPECT_TRUE(contains(message, "left_ankle_roll_joint")) << message;
  EXPECT_TRUE(
    contains(parse_error("limits:\n  estop: {position_offset: 0.3}\n", h12()),
    "'limits.estop.position_offset'"));
}

// kp_max and kd_max must be single nonnegative numbers.
TEST(Config, KpAndKdMaxAreSingleNonNegativeNumbers)
{
  const std::string list = parse_error(
    "limits:\n  clip: {kp_max: " + list_of(100.0, 27) + "}\n", h12());
  EXPECT_TRUE(contains(list, "'limits.clip.kp_max'")) << list;
  EXPECT_TRUE(contains(list, "single number")) << list;
  EXPECT_TRUE(contains(parse_error("limits:\n  clip: {kd_max: -1.0}\n", h12()), "kd_max"));
  EXPECT_TRUE(contains(parse_error("limits:\n  clip: {kp_max: -1.0}\n", h12()), "kp_max"));
  EXPECT_DOUBLE_EQ(parse("limits:\n  clip: {kp_max: 0.0}\n", h12()).limits.kp_max, 0.0);
}

// NaN, infinity and non-numbers are refused.
TEST(Config, NonFiniteNumbersAreRefused)
{
  EXPECT_TRUE(
    contains(parse_error("limits:\n  clip: {velocity_ratio: .nan}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("limits:\n  clip: {kp_max: .inf}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("stale: {upper_seconds: .nan}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("control: {publish_hz: .inf}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("control: {publish_hz: abc}\n", h12()), "number"));
}

// Topics must be absolute ROS names, and default to the documented ones.
TEST(Config, TopicsAreRosNames)
{
  const std::string message = parse_error("topics: {low_state: rt/lowstate}\n", h12());
  EXPECT_TRUE(contains(message, "'topics.low_state'")) << message;
  EXPECT_TRUE(contains(message, "/lowstate")) << message;
  EXPECT_TRUE(contains(parse_error("topics: {low_cmd_out: ''}\n", h12()), "'topics.low_cmd_out'"));

  const SafetyConfig c = parse("mode: split_mode\n", h12());
  EXPECT_EQ(c.topics.low_cmd_in, "/safety/lowcmd_in");
  EXPECT_EQ(c.topics.low_cmd_lower_in, "/safety/lowcmd_lower_in");
  EXPECT_EQ(c.topics.low_cmd_upper_in, "/safety/lowcmd_upper_in");
  EXPECT_EQ(c.topics.low_cmd_out, "/lowcmd");
  EXPECT_EQ(c.topics.low_state, "/lowstate");
  EXPECT_EQ(c.estop.triggered_topic, "/estop/triggered");
  EXPECT_EQ(c.estop.plugged_in_topic, "/estop/plugged_in");
  EXPECT_DOUBLE_EQ(c.estop.timeout_seconds, 0.5);
  EXPECT_DOUBLE_EQ(c.single_writer.handover_seconds, 0.5);
  EXPECT_EQ(c.logging.base_dir, "/tmp/golem_safety");
}

// Out-of-range scalars and unknown choices are refused; a stale time <= 0 is accepted.
TEST(Config, ScalarRangesAreChecked)
{
  EXPECT_TRUE(contains(parse_error("control: {publish_hz: 0.0}\n", h12()), "publish_hz"));
  EXPECT_TRUE(contains(parse_error("estop: {timeout_seconds: 0.0}\n", h12()),
    "estop.timeout_seconds"));
  EXPECT_TRUE(contains(parse_error("single_writer: {handover_seconds: -0.1}\n", h12()),
    "single_writer.handover_seconds"));
  EXPECT_TRUE(contains(parse_error("estop_response: {after: burst_then_silence, burst_count: 0}\n", h12()),
    "estop_response.burst_count"));
  EXPECT_TRUE(contains(parse_error("estop_response: {after: later}\n", h12()),
    "estop_response.after"));
  EXPECT_TRUE(contains(parse_error("estop_response: {mode: loud, after: burst_then_silence}\n", h12()),
    "estop_response.mode"));

  const SafetyConfig c = parse("stale: {upper_seconds: -1.0, lower_seconds: 0.0}\n", h12());
  EXPECT_DOUBLE_EQ(c.stale.upper_seconds, -1.0);
  EXPECT_DOUBLE_EQ(c.stale.lower_seconds, 0.0);
}

// single_writer.handover_seconds must be positive.
TEST(Config, HandoverSecondsMustBePositive)
{
  for (const char * value : {"0.0", "0", "-0.1"}) {
    SCOPED_TRACE(value);
    const std::string message =
      parse_error(std::string("single_writer: {handover_seconds: ") + value + "}\n", h12());
    EXPECT_TRUE(contains(message, "single_writer.handover_seconds")) << message;
    EXPECT_TRUE(contains(message, "must be positive")) << message;
  }
  EXPECT_DOUBLE_EQ(
    parse("single_writer: {handover_seconds: 0.25}\n", g1()).single_writer.handover_seconds,
    0.25);
}

// Damp requires kd, per joint, in [0, limits.clip.kd_max]; kd defaults to 0.
TEST(Config, EstopResponseDamp)
{
  EXPECT_TRUE(
    contains(parse_error("estop_response: {mode: damp, after: keep_writing}\n", g1()), "estop_response.kd"));

  const SafetyConfig damp = parse("estop_response: {mode: damp, after: keep_writing, kd: 5.0}\n", g1());
  EXPECT_EQ(damp.estop_response.mode, EstopMode::Damp);
  ASSERT_EQ(damp.estop_response.kd.size(), 29u);
  EXPECT_DOUBLE_EQ(damp.estop_response.kd[28], 5.0);

  EXPECT_TRUE(
    contains(
      parse_error("limits:\n  clip: {kd_max: 4.0}\nestop_response: {after: keep_writing, kd: 5.0}\n", g1()),
      "estop_response.kd"));
  EXPECT_TRUE(contains(parse_error("estop_response: {after: keep_writing, kd: -1.0}\n", g1()), "estop_response.kd"));

  const SafetyConfig none = parse("", h12());
  ASSERT_EQ(none.estop_response.kd.size(), 27u);
  EXPECT_DOUBLE_EQ(none.estop_response.kd[0], 0.0);
}

// The root and every section must be maps, and a YAML error names the source.
TEST(Config, RootAndSectionsMustBeMaps)
{
  EXPECT_TRUE(contains(parse_raw_error("- a\n- b\n", h12()), "mapping"));
  EXPECT_TRUE(contains(parse_raw_error("", h12()), "mapping"));
  EXPECT_TRUE(contains(parse_error("estop_response:\n", h12()), "'estop_response'"));
  EXPECT_TRUE(contains(parse_error("stale: 3\n", h12()), "'stale'"));
  EXPECT_TRUE(contains(parse_error("limits:\n  clip: 3\n", h12()), "'limits.clip'"));
  EXPECT_TRUE(contains(parse_error("mode: [a]\n", h12()), "'mode'"));
  EXPECT_TRUE(contains(parse_error("logging: {enabled: maybe}\n", h12()), "'logging.enabled'"));
  EXPECT_TRUE(contains(parse_error("a: [\n", h12()), "case.yaml"));
}

// Add the dotted path of every key under `node` to `out`, sections and their keys alike.
static void collect_keys(
  const YAML::Node & node, const std::string & prefix, std::set<std::string> & out)
{
  for (const auto & entry : node) {
    const std::string key = prefix + entry.first.as<std::string>();
    out.insert(key);
    if (entry.second.IsMap()) {
      collect_keys(entry.second, key + ".", out);
    }
  }
}

// config/example.yaml loads through the loader for every robot, names every key of the
// schema and no other, and holds the loader's default wherever a key has one.
TEST(Config, ExampleHasEveryKeyAndTheDefaults)
{
  const std::string path =
    ament_index_cpp::get_package_share_directory("golem_safety") + "/config/example.yaml";
  std::set<std::string> keys;
  collect_keys(YAML::LoadFile(path), "", keys);
  const std::set<std::string> schema = {
    "mode",
    "topics", "topics.low_cmd_in", "topics.low_cmd_lower_in", "topics.low_cmd_upper_in",
    "topics.low_cmd_out", "topics.low_state",
    "control", "control.publish_hz", "control.state_lag_ms",
    "stale", "stale.upper_seconds", "stale.lower_seconds", "stale.state_seconds",
    "single_writer", "single_writer.handover_seconds",
    "estop", "estop.triggered_topic", "estop.plugged_in_topic", "estop.timeout_seconds",
    "estop_response", "estop_response.mode", "estop_response.kd", "estop_response.after",
    "estop_response.burst_count",
    "limits", "limits.clip", "limits.clip.position_offset", "limits.clip.velocity_ratio",
    "limits.clip.torque_ratio", "limits.clip.kp_max", "limits.clip.kd_max",
    "limits.estop", "limits.estop.position_offset", "limits.estop.velocity_ratio",
    "limits.estop.torque_ratio",
    "logging", "logging.enabled", "logging.base_dir", "logging.chunk_prefix"};
  EXPECT_EQ(keys, schema);

  for (const RobotProfile * profile : {&h12(), &g1()}) {
    SCOPED_TRACE(profile->robot);
    const SafetyConfig example = golem_safety::load_safety_config(path, *profile);
    const SafetyConfig defaults = parse_raw(
      "mode: split_mode\ncontrol: {state_lag_ms: 0.0}\n"
      "estop_response: {after: burst_then_silence}\n", *profile);
    EXPECT_EQ(example.mode, defaults.mode);
    EXPECT_EQ(example.topics.low_cmd_in, defaults.topics.low_cmd_in);
    EXPECT_EQ(example.topics.low_cmd_lower_in, defaults.topics.low_cmd_lower_in);
    EXPECT_EQ(example.topics.low_cmd_upper_in, defaults.topics.low_cmd_upper_in);
    EXPECT_EQ(example.topics.low_cmd_out, defaults.topics.low_cmd_out);
    EXPECT_EQ(example.topics.low_state, defaults.topics.low_state);
    EXPECT_EQ(example.control.publish_hz, defaults.control.publish_hz);
    EXPECT_EQ(example.control.state_lag_ms, defaults.control.state_lag_ms);
    EXPECT_EQ(example.stale.upper_seconds, defaults.stale.upper_seconds);
    EXPECT_EQ(example.stale.lower_seconds, defaults.stale.lower_seconds);
    EXPECT_EQ(example.stale.state_seconds, defaults.stale.state_seconds);
    EXPECT_EQ(example.single_writer.handover_seconds, defaults.single_writer.handover_seconds);
    EXPECT_EQ(example.estop.triggered_topic, defaults.estop.triggered_topic);
    EXPECT_EQ(example.estop.plugged_in_topic, defaults.estop.plugged_in_topic);
    EXPECT_EQ(example.estop.timeout_seconds, defaults.estop.timeout_seconds);
    EXPECT_EQ(example.estop_response.mode, defaults.estop_response.mode);
    EXPECT_EQ(example.estop_response.kd, defaults.estop_response.kd);
    EXPECT_EQ(example.estop_response.after, defaults.estop_response.after);
    EXPECT_EQ(example.estop_response.burst_count, defaults.estop_response.burst_count);
    EXPECT_EQ(example.limits.q_clip_low, defaults.limits.q_clip_low);
    EXPECT_EQ(example.limits.q_clip_high, defaults.limits.q_clip_high);
    EXPECT_EQ(example.limits.dq_clip, defaults.limits.dq_clip);
    EXPECT_EQ(example.limits.tau_clip, defaults.limits.tau_clip);
    EXPECT_EQ(example.limits.q_estop_low, defaults.limits.q_estop_low);
    EXPECT_EQ(example.limits.q_estop_high, defaults.limits.q_estop_high);
    EXPECT_EQ(example.limits.dq_estop, defaults.limits.dq_estop);
    EXPECT_EQ(example.limits.tau_estop, defaults.limits.tau_estop);
    EXPECT_EQ(example.limits.kp_max, defaults.limits.kp_max);
    EXPECT_EQ(example.limits.kd_max, defaults.limits.kd_max);
    EXPECT_EQ(example.logging.enabled, defaults.logging.enabled);
    EXPECT_EQ(example.logging.base_dir, defaults.logging.base_dir);
    EXPECT_EQ(example.logging.chunk_prefix, defaults.logging.chunk_prefix);
  }
}
