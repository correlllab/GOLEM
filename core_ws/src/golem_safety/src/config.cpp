#include "golem_safety/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"

namespace golem_safety
{

namespace
{

// Defaults of the limit derivation.
constexpr double kDefaultPositionOffset = 0.01;  // rad
constexpr double kDefaultVelocityRatio = 0.5;
constexpr double kDefaultTorqueRatio = 0.5;
constexpr double kDefaultKpMax = 300.0;
constexpr double kDefaultKdMax = 30.0;

// What the error says about a key an older config may still carry.
const std::map<std::string, std::string> & replaced_keys()
{
  static const std::string estop_box = "removed: the bringup's use_estop decides";
  static const std::map<std::string, std::string> keys = {
    {"network", "removed: ROS supplies the domain, and the G1's interface is pinned in "
      "cyclonedds_local.xml"},
    {"robot", "removed: whether the robot is simulated comes from the bringup's sim argument"},
    {"engage.mode_machine", "removed: mode_machine comes from the robot profile"},
    {"estop.enabled", estop_box},
    {"estop.estop_topic", estop_box},
    {"estop.poll_hz", estop_box},
    {"estop.upper_stale_seconds", "renamed stale.upper_seconds"},
    {"estop.state_lag_ms", "moved to control.state_lag_ms"},
    {"control.lower_stale_estop_seconds", "renamed stale.lower_seconds"},
    {"control.state_stale_estop_seconds", "renamed stale.state_seconds"},
  };
  return keys;
}

struct Where
{
  const std::string & robot;
  const std::string & source;
  size_t joint_count;
};

[[noreturn]] void fail(const Where & where, const std::string & key, const std::string & what)
{
  std::string message = "safety config " + where.source + " (robot " + where.robot + "): ";
  if (!key.empty()) {
    message += "key '" + key + "': ";
  }
  throw std::runtime_error(message + what);
}

// A finite number, or a failure naming `key`.
double number_of(const Where & where, const YAML::Node & node, const std::string & key)
{
  double value = 0.0;
  try {
    if (!node.IsScalar()) {
      fail(where, key, "must be a number");
    }
    value = node.as<double>();
  } catch (const YAML::Exception &) {
    fail(where, key, "must be a number");
  }
  if (!std::isfinite(value)) {
    fail(where, key, "must be a finite number");
  }
  return value;
}

// A map in the config, addressed by its dotted path ("" for the root).
class Section
{
public:
  Section(const Where & where, const YAML::Node & node, std::string path)
  : where_(where), node_(node), path_(std::move(path))
  {
    if (!node_.IsMap()) {
      fail(where_, path_, "must be a mapping");
    }
  }

  std::string key(const std::string & name) const
  {
    return path_.empty() ? name : path_ + "." + name;
  }

  [[noreturn]] void fail_at(const std::string & name, const std::string & what) const
  {
    fail(where_, key(name), what);
  }

  bool has(const std::string & name) const {return node_[name].IsDefined();}

  // A sub-map; an absent one is empty, so every key in it takes its default.
  Section section(const std::string & name) const
  {
    return Section(where_, has(name) ? node_[name] : YAML::Node(YAML::NodeType::Map), key(name));
  }

  void allow_only(std::initializer_list<const char *> names) const
  {
    for (const auto & entry : node_) {
      std::string name;
      try {
        name = entry.first.as<std::string>();
      } catch (const YAML::Exception &) {
        fail(where_, path_, "has a key that is not a name");
      }
      if (std::find(names.begin(), names.end(), name) != names.end()) {
        continue;
      }
      std::string what = "unknown key";
      const auto replaced = replaced_keys().find(key(name));
      if (replaced != replaced_keys().end()) {
        what += " (" + replaced->second + ")";
      }
      fail(where_, key(name), what);
    }
  }

  double number(const std::string & name, double fallback) const
  {
    return has(name) ? number_of(where_, node_[name], key(name)) : fallback;
  }

  double positive(const std::string & name, double fallback) const
  {
    const double value = number(name, fallback);
    if (!(value > 0.0)) {
      fail_at(name, "must be positive");
    }
    return value;
  }

  double nonnegative(const std::string & name, double fallback) const
  {
    const double value = number(name, fallback);
    if (value < 0.0) {
      fail_at(name, "must be nonnegative");
    }
    return value;
  }

  // A number, where a per-joint list would not fit the single limit golem_safety keeps.
  double single_number(const std::string & name, double fallback) const
  {
    if (has(name) && node_[name].IsSequence()) {
      fail_at(name, "must be a single number, not a list: one limit applies to every joint");
    }
    return nonnegative(name, fallback);
  }

  int positive_integer(const std::string & name, int fallback) const
  {
    if (!has(name)) {
      return fallback;
    }
    int value = 0;
    try {
      value = node_[name].as<int>();
    } catch (const YAML::Exception &) {
      fail_at(name, "must be an integer");
    }
    if (value < 1) {
      fail_at(name, "must be at least 1");
    }
    return value;
  }

  bool boolean(const std::string & name, bool fallback) const
  {
    if (!has(name)) {
      return fallback;
    }
    try {
      return node_[name].as<bool>();
    } catch (const YAML::Exception &) {
      fail_at(name, "must be true or false");
    }
  }

  std::string text(const std::string & name, const std::string & fallback) const
  {
    if (!has(name)) {
      return fallback;
    }
    try {
      if (!node_[name].IsScalar()) {
        fail_at(name, "must be text");
      }
      return node_[name].as<std::string>();
    } catch (const YAML::Exception &) {
      fail_at(name, "must be text");
    }
  }

  // A ROS topic name: absolute, so a DDS name such as rt/lowstate is refused.
  std::string topic(const std::string & name, const std::string & fallback) const
  {
    const std::string value = text(name, fallback);
    if (value.empty() || value[0] != '/') {
      fail_at(
        name, "'" + value + "' is not a ROS topic name; it must start with '/' (the DDS topic "
        "rt/x is the ROS topic /x)");
    }
    return value;
  }

  // The value of one of `choices`, as its index.
  size_t choice(
    const std::string & name, const std::string & fallback,
    const std::vector<std::string> & choices) const
  {
    const std::string value = text(name, fallback);
    const auto found = std::find(choices.begin(), choices.end(), value);
    if (found == choices.end()) {
      fail_at(name, "'" + value + "' is not one of " + joined(choices));
    }
    return static_cast<size_t>(found - choices.begin());
  }

  // As choice, for a key with no default: leaving it out is an error.
  size_t required_choice(const std::string & name, const std::vector<std::string> & choices) const
  {
    if (!has(name)) {
      fail_at(name, "must be given, as one of " + joined(choices));
    }
    return choice(name, "", choices);
  }

  // A number for every body joint: one scalar for all, or one entry per joint.
  std::vector<double> per_joint(const std::string & name, double fallback) const
  {
    const size_t count = where_.joint_count;
    if (!has(name)) {
      return std::vector<double>(count, fallback);
    }
    const YAML::Node node = node_[name];
    if (node.IsScalar()) {
      return std::vector<double>(count, number_of(where_, node, key(name)));
    }
    if (!node.IsSequence()) {
      fail_at(name, "must be a number or a list with " + std::to_string(count) + " entries");
    }
    if (node.size() != count) {
      fail_at(
        name, "must be a number or a list with " + std::to_string(count) +
        " entries, one per body joint of robot " + where_.robot + ", not " +
        std::to_string(node.size()));
    }
    std::vector<double> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(number_of(where_, node[i], key(name) + "[" + std::to_string(i) + "]"));
    }
    return values;
  }

private:
  static std::string joined(const std::vector<std::string> & choices)
  {
    std::string list;
    for (const auto & choice : choices) {
      list += (list.empty() ? "" : ", ") + choice;
    }
    return list;
  }

  const Where & where_;
  YAML::Node node_;
  std::string path_;
};

struct LimitSet
{
  std::vector<double> q_low, q_high, dq, tau;
};

// The per-slot limits one `limits.clip` or `limits.estop` section derives from the
// joint table.
LimitSet derive_limits(const RobotProfile & profile, const Section & policy)
{
  const std::vector<double> offset = policy.per_joint("position_offset", kDefaultPositionOffset);
  const std::vector<double> velocity_ratio =
    policy.per_joint("velocity_ratio", kDefaultVelocityRatio);
  const std::vector<double> torque_ratio = policy.per_joint("torque_ratio", kDefaultTorqueRatio);

  LimitSet out;
  for (size_t i = 0; i < profile.joints.size(); ++i) {
    const JointLimit & joint = profile.joints[i];
    const double low = joint.q_low + offset[i];
    const double high = joint.q_high - offset[i];
    if (low >= high) {
      policy.fail_at(
        "position_offset", "too large for joint " + std::to_string(i) + " " + joint.name);
    }
    out.q_low.push_back(low);
    out.q_high.push_back(high);
    out.dq.push_back(joint.dq_max * velocity_ratio[i]);
    out.tau.push_back(joint.tau_max * torque_ratio[i]);
  }
  return out;
}

// The section holding the clip policy: limits.clip, or its alias limits.enforce.
Section clip_section(const Section & limits)
{
  if (limits.has("clip") && limits.has("enforce")) {
    limits.fail_at("enforce", "is an alias of limits.clip; give only one of them");
  }
  return limits.section(limits.has("enforce") ? "enforce" : "clip");
}

// kp_max and kd_max, each under its own name or its alias (kp_abs, kd_abs).
double gain_limit(
  const Section & clip, const std::string & name, const std::string & alias, double fallback)
{
  if (clip.has(name) && clip.has(alias)) {
    clip.fail_at(alias, "is an alias of " + name + "; give only one of them");
  }
  return clip.single_number(clip.has(name) ? name : alias, fallback);
}

DerivedLimits derive_all(const RobotProfile & profile, const Section & limits)
{
  const Section clip = clip_section(limits);
  clip.allow_only(
    {"position_offset", "velocity_ratio", "torque_ratio", "kp_max", "kd_max", "kp_abs", "kd_abs"});
  const Section estop = limits.section("estop");
  estop.allow_only({"position_offset", "velocity_ratio", "torque_ratio"});

  DerivedLimits out;
  const LimitSet clipped = derive_limits(profile, clip);
  out.q_clip_low = clipped.q_low;
  out.q_clip_high = clipped.q_high;
  out.dq_clip = clipped.dq;
  out.tau_clip = clipped.tau;
  out.kp_max = gain_limit(clip, "kp_max", "kp_abs", kDefaultKpMax);
  out.kd_max = gain_limit(clip, "kd_max", "kd_abs", kDefaultKdMax);

  const LimitSet estopped = derive_limits(profile, estop);
  out.q_estop_low = estopped.q_low;
  out.q_estop_high = estopped.q_high;
  out.dq_estop = estopped.dq;
  out.tau_estop = estopped.tau;
  return out;
}

EstopResponse parse_estop_response(
  const Section & section, Mode config_mode, const DerivedLimits & limits)
{
  section.allow_only({"mode", "kd", "after", "burst_count"});
  EstopResponse out;
  const size_t mode = section.choice("mode", "zero", {"zero", "damp"});
  out.mode = mode == 0 ? EstopMode::Zero : EstopMode::Damp;
  const size_t after =
    section.required_choice("after", {"burst_then_silence", "keep_writing"});
  out.after = after == 0 ? EstopAfter::BurstThenSilence : EstopAfter::KeepWriting;
  if (config_mode == Mode::ArmSdk && out.after != EstopAfter::KeepWriting) {
    section.fail_at(
      "after", "must be keep_writing in arm_sdk_mode: the weight ramp keeps writing after "
      "an estop");
  }
  out.burst_count = section.positive_integer("burst_count", 50);
  out.kd = section.per_joint("kd", 0.0);
  for (const double kd : out.kd) {
    if (kd < 0.0) {
      section.fail_at("kd", "must be nonnegative");
    }
    if (kd > limits.kd_max) {
      section.fail_at("kd", "must not exceed limits.clip.kd_max");
    }
  }
  if (out.mode == EstopMode::Damp && !section.has("kd")) {
    section.fail_at("kd", "is required when estop_response.mode is damp");
  }
  return out;
}

SafetyConfig parse_checked(
  const YAML::Node & root_node, const std::string & source, const RobotProfile & profile)
{
  const Where where{profile.robot, source, profile.joints.size()};
  const Section root(where, root_node, "");
  root.allow_only(
    {"mode", "topics", "control", "stale", "single_writer", "startup", "engage", "arm_sdk",
      "estop", "estop_response", "limits", "logging"});

  SafetyConfig config;
  config.robot = profile.robot;
  config.source = source;

  // A config without a mode runs the profile's first mode.
  const std::string mode_text = root.text("mode", mode_name(profile.modes.front()));
  try {
    config.mode = parse_mode(mode_text);
  } catch (const std::runtime_error & e) {
    root.fail_at("mode", e.what());
  }
  if (std::find(profile.modes.begin(), profile.modes.end(), config.mode) == profile.modes.end()) {
    std::string modes;
    for (const Mode mode : profile.modes) {
      modes += std::string(modes.empty() ? "" : ", ") + mode_name(mode);
    }
    root.fail_at(
      "mode", "'" + mode_text + "' is not a mode of robot " + profile.robot + " (" + modes + ")");
  }

  const Section topics = root.section("topics");
  topics.allow_only({"low_cmd_in", "low_cmd_lower_in", "low_cmd_upper_in", "low_cmd_out",
      "low_state"});
  config.topics.low_cmd_in = topics.topic("low_cmd_in", "/safety/lowcmd_in");
  config.topics.low_cmd_lower_in = topics.topic("low_cmd_lower_in", "/safety/lowcmd_lower_in");
  config.topics.low_cmd_upper_in = topics.topic("low_cmd_upper_in", "/safety/lowcmd_upper_in");
  config.topics.low_cmd_out =
    topics.topic("low_cmd_out", config.mode == Mode::ArmSdk ? "/arm_sdk" : "/lowcmd");
  config.topics.low_state = topics.topic("low_state", "/lowstate");

  const Section control = root.section("control");
  control.allow_only({"publish_hz", "state_lag_ms"});
  config.control.publish_hz = control.positive("publish_hz", config.control.publish_hz);
  config.control.state_lag_ms = control.number("state_lag_ms", config.control.state_lag_ms);

  const Section stale = root.section("stale");
  stale.allow_only({"upper_seconds", "lower_seconds", "state_seconds"});
  config.stale.upper_seconds = stale.number("upper_seconds", config.stale.upper_seconds);
  config.stale.lower_seconds = stale.number("lower_seconds", config.stale.lower_seconds);
  config.stale.state_seconds = stale.number("state_seconds", config.stale.state_seconds);

  const Section single_writer = root.section("single_writer");
  single_writer.allow_only({"handover_seconds"});
  config.single_writer.handover_seconds =
    single_writer.nonnegative("handover_seconds", config.single_writer.handover_seconds);

  const Section startup = root.section("startup");
  startup.allow_only({"rule"});
  const size_t rule = startup.required_choice("rule", {"zero_stream", "engage_gate"});
  config.startup.rule = rule == 0 ? StartupRule::ZeroStream : StartupRule::EngageGate;

  const Section engage = root.section("engage");
  engage.allow_only({"max_state_age_seconds", "max_position_error"});
  config.engage.max_state_age_seconds =
    engage.positive("max_state_age_seconds", config.engage.max_state_age_seconds);
  config.engage.max_position_error =
    engage.positive("max_position_error", config.engage.max_position_error);

  const Section arm_sdk = root.section("arm_sdk");
  arm_sdk.allow_only({"engage_seconds", "release_seconds", "waist_kp_min"});
  config.arm_sdk.engage_seconds =
    arm_sdk.nonnegative("engage_seconds", config.arm_sdk.engage_seconds);
  config.arm_sdk.release_seconds =
    arm_sdk.nonnegative("release_seconds", config.arm_sdk.release_seconds);
  config.arm_sdk.waist_kp_min = arm_sdk.nonnegative("waist_kp_min", config.arm_sdk.waist_kp_min);

  const Section estop = root.section("estop");
  estop.allow_only({"triggered_topic", "plugged_in_topic", "timeout_seconds"});
  config.estop.triggered_topic = estop.topic("triggered_topic", config.estop.triggered_topic);
  config.estop.plugged_in_topic = estop.topic("plugged_in_topic", config.estop.plugged_in_topic);
  config.estop.timeout_seconds = estop.positive("timeout_seconds", config.estop.timeout_seconds);

  const Section limits = root.section("limits");
  limits.allow_only({"clip", "enforce", "estop"});
  config.limits = derive_all(profile, limits);

  config.estop_response =
    parse_estop_response(root.section("estop_response"), config.mode, config.limits);

  // arm_sdk_mode rejects a command whose waist kp is below waist_kp_min, so the
  // clip must allow it. kp_max is one number for every joint, the waist included.
  if (config.mode == Mode::ArmSdk && config.arm_sdk.waist_kp_min > config.limits.kp_max) {
    arm_sdk.fail_at("waist_kp_min", "must not exceed limits.clip.kp_max");
  }

  const Section logging = root.section("logging");
  logging.allow_only({"enabled", "base_dir", "chunk_prefix"});
  config.logging.enabled = logging.boolean("enabled", config.logging.enabled);
  config.logging.base_dir = logging.text("base_dir", config.logging.base_dir);
  config.logging.chunk_prefix = logging.text("chunk_prefix", config.logging.chunk_prefix);
  return config;
}

bool ends_with(const std::string & text, const std::string & suffix)
{
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

SafetyConfig parse_safety_config(
  const std::string & yaml_text, const std::string & source, const RobotProfile & profile)
{
  const Where where{profile.robot, source, profile.joints.size()};
  try {
    return parse_checked(YAML::Load(yaml_text), source, profile);
  } catch (const YAML::Exception & e) {
    fail(where, "", e.what());
  }
}

SafetyConfig load_safety_config(const std::string & name_or_path, const RobotProfile & profile)
{
  std::string path = name_or_path;
  if (name_or_path.find('/') == std::string::npos) {
    try {
      path = ament_index_cpp::get_package_share_directory("golem_safety") + "/config/" +
        profile.robot + "/" + name_or_path + (ends_with(name_or_path, ".yaml") ? "" : ".yaml");
    } catch (const std::exception & e) {
      throw std::runtime_error(
              "cannot load the config '" + name_or_path + "' of robot " + profile.robot + ": " +
              e.what());
    }
  }
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error(
            "no config '" + name_or_path + "' for robot " + profile.robot + ": cannot read " +
            path);
  }
  std::ostringstream text;
  text << file.rdbuf();
  return parse_safety_config(text.str(), path, profile);
}

}  // namespace golem_safety
