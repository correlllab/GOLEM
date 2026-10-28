#include <gtest/gtest.h>

#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"

using golem_safety::EstopAfter;
using golem_safety::EstopMode;
using golem_safety::Mode;
using golem_safety::RobotProfile;
using golem_safety::SafetyConfig;

namespace {

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

const char * const kH12Configs[] = {
  "default_safety_full", "default_safety_split", "locomotion_safety_full", "relax_safety_full",
  "relax_safety_split", "sim_safety_split", "tight_safety_full", "tight_safety_split"};

const char * const kG1Configs[] = {"full", "sim_split", "split"};

// The message of the std::runtime_error that `fn` throws; fails the test when
// it throws nothing.
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

bool contains(const std::string & text, const std::string & part)
{
  return text.find(part) != std::string::npos;
}

// A config file in a private temp directory, removed with the object.
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

// estop_response.after and control.state_lag_ms have no default, so most tests that are
// not about them leave them to this: `text` plus whichever of the two sections it does not
// give, with the robot's carried values.
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

SafetyConfig parse_raw(const std::string & text, const RobotProfile & profile)
{
  return golem_safety::parse_safety_config(text, "case.yaml", profile);
}

std::string parse_raw_error(const std::string & text, const RobotProfile & profile)
{
  return error_of([&] {parse_raw(text, profile);});
}

SafetyConfig parse(const std::string & text, const RobotProfile & profile)
{
  return parse_raw(completed(text, profile), profile);
}

std::string parse_error(const std::string & text, const RobotProfile & profile)
{
  return error_of([&] {parse(text, profile);});
}

std::string list_of(double value, size_t count)
{
  std::string text = "[";
  for (size_t i = 0; i < count; ++i) {
    text += (i ? ", " : "") + std::to_string(value);
  }
  return text + "]";
}

}  // namespace

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

  // A path with a directory is used as given.
  const TempConfig file(completed("mode: split_mode\n", h12()));
  EXPECT_EQ(golem_safety::load_safety_config(file.path(), h12()).source, file.path());

  // A name the robot lacks reports the robot and the file it looked for.
  const std::string message =
    error_of([] {golem_safety::load_safety_config("split", h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, "/config/h12/split.yaml")) << message;
}

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

  // The estop-section keys are not valid in the estop limits.
  EXPECT_TRUE(
    contains(parse_error("limits:\n  estop:\n    kp_max: 1.0\n", h12()), "'limits.estop.kp_max'"));
}

TEST(Config, RemovedAndRenamedKeysNameTheirReplacement)
{
  EXPECT_TRUE(contains(parse_error("network: {domain_id: 1}\n", h12()), "removed"));
  EXPECT_TRUE(contains(parse_error("robot: real\n", g1()), "sim"));
  EXPECT_TRUE(contains(parse_error("estop: {enabled: true}\n", h12()), "use_estop"));
  EXPECT_TRUE(contains(parse_error("estop: {poll_hz: 500.0}\n", h12()), "use_estop"));
  EXPECT_TRUE(
    contains(parse_error("estop: {upper_stale_seconds: 2.0}\n", h12()), "stale.upper_seconds"));
  EXPECT_TRUE(
    contains(parse_error("estop: {state_lag_ms: 100.0}\n", h12()), "control.state_lag_ms"));
  EXPECT_TRUE(
    contains(
      parse_error("control: {lower_stale_estop_seconds: 0.2}\n", g1()), "stale.lower_seconds"));
  EXPECT_TRUE(
    contains(
      parse_error("control: {state_stale_estop_seconds: 0.1}\n", g1()), "stale.state_seconds"));
}

TEST(Config, ArmSdkModeRefused)
{
  // The G1 runs split_mode or full_body_mode: the message names the mode it was given and the
  // robot's own.
  const std::string message = parse_error("mode: arm_sdk_mode\n", g1());
  EXPECT_TRUE(contains(message, "(robot g1)")) << message;
  EXPECT_TRUE(contains(message, "'mode'")) << message;
  EXPECT_TRUE(
    contains(message, "'arm_sdk_mode' is not a mode of robot g1 (split_mode, full_body_mode)")) <<
    message;
}

TEST(Config, ArmSdkKeysRefused)
{
  const std::string reason =
    "unknown key (arm_sdk mode was removed: the G1 runs split_mode or full_body_mode)";
  for (const char * section : {
      "arm_sdk: {engage_seconds: 1.0, release_seconds: 3.0}\n", "arm_sdk: {waist_kp_min: 40.0}\n",
      "arm_sdk: {}\n"})
  {
    SCOPED_TRACE(section);
    const std::string message = parse_error(std::string("mode: split_mode\n") + section, g1());
    EXPECT_TRUE(
      contains(message, "safety config case.yaml (robot g1): key 'arm_sdk': " + reason)) <<
      message;
  }
  // The H1-2's configs are refused the same way.
  const std::string h12_message = parse_error("arm_sdk: {}\n", h12());
  EXPECT_TRUE(contains(h12_message, "key 'arm_sdk': " + reason)) << h12_message;
}

TEST(Config, StartupAndEngageKeysRefused)
{
  // Every robot streams zero gains from boot, so neither section has a choice left to make.
  const std::string reason =
    "unknown key (removed: every robot streams zero gains from boot and engages each input "
    "on its own first valid command)";
  struct Case
  {
    const RobotProfile & profile;
    std::string rest;  // the required keys, with the robot's carried values
  };
  const Case cases[] = {
    {h12(), "control: {state_lag_ms: 100.0}\nestop_response: {after: burst_then_silence}\n"},
    {g1(), "control: {state_lag_ms: 0.0}\nestop_response: {after: keep_writing}\n"},
  };
  const std::pair<const char *, const char *> sections[] = {
    {"startup", "startup: {rule: zero_stream}\n"},
    {"startup", "startup: {rule: engage_gate}\n"},
    {"startup", "startup: {}\n"},
    {"engage", "engage: {max_state_age_seconds: 0.1, max_position_error: 0.15}\n"},
    {"engage", "engage: {mode_machine: 5}\n"},
    {"engage", "engage: {}\n"},
  };
  for (const Case & c : cases) {
    for (const auto & [key, section] : sections) {
      SCOPED_TRACE(c.profile.robot + ": " + section);
      const std::string message =
        parse_raw_error("mode: split_mode\n" + c.rest + section, c.profile);
      EXPECT_TRUE(
        contains(
          message, "safety config case.yaml (robot " + c.profile.robot + "): key '" + key +
          "': " + reason)) << message;
    }
  }
}

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

  // Both robots have both.
  for (const RobotProfile * profile : {&h12(), &g1()}) {
    EXPECT_EQ(parse("mode: split_mode\n", *profile).mode, Mode::Split);
    EXPECT_EQ(parse("mode: full_body_mode\n", *profile).mode, Mode::FullBody);
  }
}

TEST(Config, ModeDefaultsToTheProfilesFirstMode)
{
  EXPECT_EQ(parse("", h12()).mode, Mode::FullBody);
  EXPECT_EQ(parse("", g1()).mode, Mode::Split);
}

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

  // A list for the other robot's joint count is just as wrong.
  message = parse_error("limits:\n  estop:\n    torque_ratio: " + list_of(0.5, 29) + "\n", h12());
  EXPECT_TRUE(contains(message, "'limits.estop.torque_ratio'")) << message;
  message = parse_error("limits:\n  clip:\n    position_offset: " + list_of(0.0, 27) + "\n", g1());
  EXPECT_TRUE(contains(message, "'limits.clip.position_offset'")) << message;
  EXPECT_TRUE(contains(message, "29")) << message;

  // One entry per joint is accepted and applies joint by joint.
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

TEST(Config, StateLagMsIsRequired)
{
  // Everything but control.state_lag_ms.
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

  // The same through a file: the message names the file.
  const TempConfig file("mode: split_mode\n" + h12_rest);
  message = error_of([&] {golem_safety::load_safety_config(file.path(), h12());});
  EXPECT_TRUE(contains(message, file.path())) << message;
  EXPECT_TRUE(contains(message, "'control.state_lag_ms'")) << message;
}

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
  // burst_count is the one key here that keeps a default.
  EXPECT_EQ(parse("", h12()).estop_response.burst_count, 50);
}

TEST(Config, H12UpperStaleSecondsAreTheOldEffectiveValues)
{
  // default_safety_split and tight_safety_split had no key, so the old loader's 2.0 applied.
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("default_safety_split", h12())
    .stale.upper_seconds, 2.0);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("tight_safety_split", h12())
    .stale.upper_seconds, 2.0);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("relax_safety_split", h12())
    .stale.upper_seconds, 0.0);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("sim_safety_split", h12())
    .stale.upper_seconds, 0.0);
}

TEST(Config, G1ConfigsKeepWriting)
{
  for (const char * name : kG1Configs) {
    const SafetyConfig config = golem_safety::load_safety_config(name, g1());
    EXPECT_EQ(config.estop_response.after, EstopAfter::KeepWriting) << name;
    EXPECT_EQ(config.estop_response.mode, EstopMode::Zero) << name;
    EXPECT_TRUE(config.logging.enabled) << name;
    EXPECT_EQ(config.logging.base_dir, "/tmp/golem_safety") << name;
    // The G1's lag check is off.
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

TEST(Config, EveryCarriedConfigLoads)
{
  for (const char * name : kH12Configs) {
    EXPECT_NO_THROW(golem_safety::load_safety_config(name, h12())) << name;
  }
  for (const char * name : kG1Configs) {
    EXPECT_NO_THROW(golem_safety::load_safety_config(name, g1())) << name;
  }
}

TEST(Config, ConfigNamesBelongToTheirRobot)
{
  // The H1-2's names do not exist for the G1 and the other way round.
  for (const char * name : kH12Configs) {
    EXPECT_THROW(golem_safety::load_safety_config(name, g1()), std::runtime_error) << name;
  }
  for (const char * name : kG1Configs) {
    EXPECT_THROW(golem_safety::load_safety_config(name, h12()), std::runtime_error) << name;
  }
}

TEST(Config, PathInAnotherRobotsDirectoryIsRefused)
{
  // Several configs of one robot parse for the other (the G1's full.yaml for the H1-2, the
  // H1-2's sim_safety_split.yaml for the G1), so a path under another robot's config
  // directory is refused by that directory's name.
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
  // The robot's own directory, given as a path, and a directory of any other name load.
  EXPECT_NO_THROW(
    golem_safety::load_safety_config(share + "/config/h12/sim_safety_split.yaml", h12()));
  const TempConfig file(completed("mode: split_mode\n", h12()));
  EXPECT_NO_THROW(golem_safety::load_safety_config(file.path(), h12()));
}

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

TEST(Config, EnforceAndKpAbsKdAbsAreAliases)
{
  const SafetyConfig c = parse(
    "mode: full_body_mode\n"
    "limits:\n"
    "  enforce: {position_offset: 0.02, velocity_ratio: 0.2, torque_ratio: 0.3,"
    " kp_abs: 111.0, kd_abs: 7.0}\n", h12());
  EXPECT_DOUBLE_EQ(c.limits.q_clip_low[0], h12().joints[0].q_low + 0.02);
  EXPECT_DOUBLE_EQ(c.limits.dq_clip[0], h12().joints[0].dq_max * 0.2);
  EXPECT_DOUBLE_EQ(c.limits.tau_clip[0], h12().joints[0].tau_max * 0.3);
  EXPECT_DOUBLE_EQ(c.limits.kp_max, 111.0);
  EXPECT_DOUBLE_EQ(c.limits.kd_max, 7.0);
  // The estop limits keep their own defaults.
  EXPECT_DOUBLE_EQ(c.limits.q_estop_low[0], h12().joints[0].q_low + 0.01);

  // An alias next to its own key is ambiguous.
  EXPECT_TRUE(
    contains(
      parse_error("limits:\n  clip: {kp_max: 1.0, kp_abs: 2.0}\n", h12()), "limits.clip.kp_abs"));
  EXPECT_TRUE(
    contains(
      parse_error("limits:\n  clip: {}\n  enforce: {}\n", h12()), "limits.enforce"));
}

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

TEST(Config, NonFiniteNumbersAreRefused)
{
  EXPECT_TRUE(
    contains(parse_error("limits:\n  clip: {velocity_ratio: .nan}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("limits:\n  clip: {kp_max: .inf}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("stale: {upper_seconds: .nan}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("control: {publish_hz: .inf}\n", h12()), "finite"));
  EXPECT_TRUE(contains(parse_error("control: {publish_hz: abc}\n", h12()), "number"));
}

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

  // A value <= 0 disables a stale report, so a negative one is accepted.
  const SafetyConfig c = parse("stale: {upper_seconds: -1.0, lower_seconds: 0.0}\n", h12());
  EXPECT_DOUBLE_EQ(c.stale.upper_seconds, -1.0);
  EXPECT_DOUBLE_EQ(c.stale.lower_seconds, 0.0);
}

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

TEST(Config, EstopResponseDamp)
{
  EXPECT_TRUE(
    contains(parse_error("estop_response: {mode: damp, after: keep_writing}\n", g1()), "estop_response.kd"));

  const SafetyConfig damp = parse("estop_response: {mode: damp, after: keep_writing, kd: 5.0}\n", g1());
  EXPECT_EQ(damp.estop_response.mode, EstopMode::Damp);
  ASSERT_EQ(damp.estop_response.kd.size(), 29u);
  EXPECT_DOUBLE_EQ(damp.estop_response.kd[28], 5.0);

  // kd stays within what the clip allows a controller.
  EXPECT_TRUE(
    contains(
      parse_error("limits:\n  clip: {kd_max: 4.0}\nestop_response: {after: keep_writing, kd: 5.0}\n", g1()),
      "estop_response.kd"));
  EXPECT_TRUE(contains(parse_error("estop_response: {after: keep_writing, kd: -1.0}\n", g1()), "estop_response.kd"));

  const SafetyConfig none = parse("", h12());
  ASSERT_EQ(none.estop_response.kd.size(), 27u);
  EXPECT_DOUBLE_EQ(none.estop_response.kd[0], 0.0);
}

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
