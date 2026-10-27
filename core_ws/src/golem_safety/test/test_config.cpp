#include <gtest/gtest.h>

#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"

using golem_safety::EstopAfter;
using golem_safety::EstopMode;
using golem_safety::Mode;
using golem_safety::RobotProfile;
using golem_safety::SafetyConfig;
using golem_safety::StartupRule;

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

const char * const kG1Configs[] = {"arm_sdk", "full", "sim_arm_sdk", "sim_split", "split"};

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

SafetyConfig parse(const std::string & text, const RobotProfile & profile)
{
  return golem_safety::parse_safety_config(text, "case.yaml", profile);
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

  const SafetyConfig arm_sdk = golem_safety::load_safety_config("arm_sdk", g1());
  EXPECT_EQ(arm_sdk.robot, "g1");
  EXPECT_EQ(arm_sdk.mode, Mode::ArmSdk);
  EXPECT_TRUE(contains(arm_sdk.source, "/config/g1/arm_sdk.yaml")) << arm_sdk.source;

  // A path with a directory is used as given.
  const TempConfig file("mode: split_mode\n");
  EXPECT_EQ(golem_safety::load_safety_config(file.path(), h12()).source, file.path());

  // A name the robot lacks reports the robot and the file it looked for.
  const std::string message =
    error_of([] {golem_safety::load_safety_config("arm_sdk", h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, "/config/h12/arm_sdk.yaml")) << message;
}

TEST(Config, UnknownKeyThrows)
{
  const TempConfig top("mode: split_mode\nbogus: 1\n");
  std::string message = error_of([&] {golem_safety::load_safety_config(top.path(), h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, top.path())) << message;
  EXPECT_TRUE(contains(message, "'bogus'")) << message;

  const TempConfig nested("mode: split_mode\nlimits:\n  clip:\n    kp_maxx: 1.0\n");
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
  EXPECT_TRUE(contains(parse_error("engage: {mode_machine: 5}\n", g1()), "profile"));
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

TEST(Config, ModeNotInProfileThrows)
{
  const TempConfig file("mode: arm_sdk_mode\n");
  const std::string message =
    error_of([&] {golem_safety::load_safety_config(file.path(), h12());});
  EXPECT_TRUE(contains(message, "h12")) << message;
  EXPECT_TRUE(contains(message, file.path())) << message;
  EXPECT_TRUE(contains(message, "'mode'")) << message;
  EXPECT_TRUE(contains(message, "arm_sdk_mode")) << message;

  EXPECT_TRUE(contains(parse_error("mode: bogus_mode\n", h12()), "'mode'"));

  // The G1 has all three.
  EXPECT_EQ(parse("mode: arm_sdk_mode\n", g1()).mode, Mode::ArmSdk);
  EXPECT_EQ(parse("mode: split_mode\n", g1()).mode, Mode::Split);
  EXPECT_EQ(parse("mode: full_body_mode\n", g1()).mode, Mode::FullBody);
}

TEST(Config, ModeDefaultsToTheProfilesFirstMode)
{
  EXPECT_EQ(parse("{}", h12()).mode, Mode::FullBody);
  EXPECT_EQ(parse("{}", g1()).mode, Mode::ArmSdk);
}

TEST(Config, ListLengthMustMatchJoints)
{
  const TempConfig short_list(
    "mode: split_mode\nlimits:\n  clip:\n    velocity_ratio: [0.1, 0.1]\n");
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
    "estop_response: {mode: damp, kd: " + list_of(1.0, 3) + "}\n", g1());
  EXPECT_TRUE(contains(message, "'estop_response.kd'")) << message;
}

TEST(Config, H12DefaultsAreZeroStreamAndBurst)
{
  // Keys left out take the schema defaults.
  const SafetyConfig minimal = parse("mode: split_mode\n", h12());
  EXPECT_EQ(minimal.startup.rule, StartupRule::ZeroStream);
  EXPECT_EQ(minimal.estop_response.mode, EstopMode::Zero);
  EXPECT_EQ(minimal.estop_response.after, EstopAfter::BurstThenSilence);
  EXPECT_EQ(minimal.estop_response.burst_count, 50);

  // Every carried H1-2 config says so itself.
  for (const char * name : kH12Configs) {
    const SafetyConfig config = golem_safety::load_safety_config(name, h12());
    EXPECT_EQ(config.startup.rule, StartupRule::ZeroStream) << name;
    EXPECT_EQ(config.estop_response.mode, EstopMode::Zero) << name;
    EXPECT_EQ(config.estop_response.after, EstopAfter::BurstThenSilence) << name;
    EXPECT_EQ(config.estop_response.burst_count, 50) << name;
    EXPECT_DOUBLE_EQ(config.stale.lower_seconds, 0.2) << name;
    EXPECT_DOUBLE_EQ(config.stale.state_seconds, 0.1) << name;
    EXPECT_DOUBLE_EQ(config.control.state_lag_ms, 100.0) << name;
    EXPECT_DOUBLE_EQ(config.control.publish_hz, 500.0) << name;
    EXPECT_FALSE(config.logging.enabled) << name;
    EXPECT_EQ(config.logging.base_dir, "/tmp/h12_safety_layer") << name;
    EXPECT_EQ(config.logging.chunk_prefix, "chunk_") << name;
    EXPECT_EQ(config.topics.low_cmd_out, "/lowcmd") << name;
    EXPECT_EQ(config.topics.low_state, "/lowstate") << name;
  }
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

TEST(Config, G1ConfigsUseTheEngageGateAndKeepWriting)
{
  for (const char * name : kG1Configs) {
    const SafetyConfig config = golem_safety::load_safety_config(name, g1());
    EXPECT_EQ(config.startup.rule, StartupRule::EngageGate) << name;
    EXPECT_EQ(config.estop_response.after, EstopAfter::KeepWriting) << name;
    EXPECT_EQ(config.estop_response.mode, EstopMode::Zero) << name;
    EXPECT_TRUE(config.logging.enabled) << name;
    EXPECT_EQ(config.logging.base_dir, "/tmp/g1_safety_layer") << name;
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

  const SafetyConfig arm_sdk = golem_safety::load_safety_config("arm_sdk", g1());
  EXPECT_EQ(arm_sdk.topics.low_cmd_out, "/arm_sdk");
  EXPECT_EQ(arm_sdk.topics.low_cmd_upper_in, "/safety/lowcmd_upper_in");
  EXPECT_DOUBLE_EQ(arm_sdk.arm_sdk.engage_seconds, 1.0);
  EXPECT_DOUBLE_EQ(arm_sdk.arm_sdk.release_seconds, 3.0);
  EXPECT_DOUBLE_EQ(arm_sdk.arm_sdk.waist_kp_min, 40.0);

  const SafetyConfig sim = golem_safety::load_safety_config("sim_arm_sdk", g1());
  EXPECT_DOUBLE_EQ(sim.arm_sdk.release_seconds, 1.0);
  EXPECT_DOUBLE_EQ(sim.engage.max_state_age_seconds, 1.0);
  EXPECT_DOUBLE_EQ(sim.engage.max_position_error, 0.15);
  EXPECT_DOUBLE_EQ(golem_safety::load_safety_config("sim_split", g1()).stale.lower_seconds, 1.0);
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
  EXPECT_EQ(parse("{}", g1()).limits.q_clip_low.size(), 29u);
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
  EXPECT_TRUE(contains(parse_error("engage: {max_state_age_seconds: 0.0}\n", g1()),
    "engage.max_state_age_seconds"));
  EXPECT_TRUE(contains(parse_error("engage: {max_position_error: -1.0}\n", g1()),
    "engage.max_position_error"));
  EXPECT_TRUE(contains(parse_error("arm_sdk: {engage_seconds: -1.0}\n", g1()),
    "arm_sdk.engage_seconds"));
  EXPECT_TRUE(contains(parse_error("arm_sdk: {release_seconds: -1.0}\n", g1()),
    "arm_sdk.release_seconds"));
  EXPECT_TRUE(contains(parse_error("estop: {timeout_seconds: 0.0}\n", h12()),
    "estop.timeout_seconds"));
  EXPECT_TRUE(contains(parse_error("single_writer: {handover_seconds: -0.1}\n", h12()),
    "single_writer.handover_seconds"));
  EXPECT_TRUE(contains(parse_error("estop_response: {burst_count: 0}\n", h12()),
    "estop_response.burst_count"));
  EXPECT_TRUE(contains(parse_error("startup: {rule: sometimes}\n", h12()), "startup.rule"));
  EXPECT_TRUE(contains(parse_error("estop_response: {after: later}\n", h12()),
    "estop_response.after"));
  EXPECT_TRUE(contains(parse_error("estop_response: {mode: loud}\n", h12()),
    "estop_response.mode"));

  // A value <= 0 disables a stale report, so a negative one is accepted.
  const SafetyConfig c = parse("stale: {upper_seconds: -1.0, lower_seconds: 0.0}\n", h12());
  EXPECT_DOUBLE_EQ(c.stale.upper_seconds, -1.0);
  EXPECT_DOUBLE_EQ(c.stale.lower_seconds, 0.0);
}

TEST(Config, EstopResponseDamp)
{
  EXPECT_TRUE(
    contains(parse_error("estop_response: {mode: damp}\n", g1()), "estop_response.kd"));

  const SafetyConfig damp = parse("estop_response: {mode: damp, kd: 5.0}\n", g1());
  EXPECT_EQ(damp.estop_response.mode, EstopMode::Damp);
  ASSERT_EQ(damp.estop_response.kd.size(), 29u);
  EXPECT_DOUBLE_EQ(damp.estop_response.kd[28], 5.0);

  // kd stays within what the clip allows a controller.
  EXPECT_TRUE(
    contains(
      parse_error("limits:\n  clip: {kd_max: 4.0}\nestop_response: {kd: 5.0}\n", g1()),
      "estop_response.kd"));
  EXPECT_TRUE(contains(parse_error("estop_response: {kd: -1.0}\n", g1()), "estop_response.kd"));

  const SafetyConfig none = parse("{}", h12());
  ASSERT_EQ(none.estop_response.kd.size(), 27u);
  EXPECT_DOUBLE_EQ(none.estop_response.kd[0], 0.0);
}

TEST(Config, WaistKpMinMayNotExceedKpMax)
{
  const std::string message =
    parse_error("arm_sdk: {waist_kp_min: 400.0}\n", g1());
  EXPECT_TRUE(contains(message, "'arm_sdk.waist_kp_min'")) << message;
  EXPECT_TRUE(contains(message, "g1")) << message;
  EXPECT_NO_THROW(parse("arm_sdk: {waist_kp_min: 300.0}\n", g1()));

  // Only arm_sdk_mode uses waist_kp_min, so the other modes do not check it.
  EXPECT_NO_THROW(
    parse("mode: split_mode\nlimits:\n  clip: {kp_max: 30.0}\n", g1()));
  EXPECT_TRUE(
    contains(
      parse_error("limits:\n  clip: {kp_max: 30.0}\n", g1()), "'arm_sdk.waist_kp_min'"));
  EXPECT_TRUE(contains(parse_error("arm_sdk: {waist_kp_min: -1.0}\n", g1()),
    "arm_sdk.waist_kp_min"));
}

TEST(Config, RootAndSectionsMustBeMaps)
{
  EXPECT_TRUE(contains(parse_error("- a\n- b\n", h12()), "mapping"));
  EXPECT_TRUE(contains(parse_error("", h12()), "mapping"));
  EXPECT_TRUE(contains(parse_error("estop_response:\n", h12()), "'estop_response'"));
  EXPECT_TRUE(contains(parse_error("stale: 3\n", h12()), "'stale'"));
  EXPECT_TRUE(contains(parse_error("limits:\n  clip: 3\n", h12()), "'limits.clip'"));
  EXPECT_TRUE(contains(parse_error("mode: [a]\n", h12()), "'mode'"));
  EXPECT_TRUE(contains(parse_error("logging: {enabled: maybe}\n", h12()), "'logging.enabled'"));
  EXPECT_TRUE(contains(parse_error("a: [\n", h12()), "case.yaml"));
}
