#include "golem_safety/robot_profile.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "ament_index_cpp/get_package_share_directory.hpp"

namespace golem_safety
{

Mode parse_mode(const std::string & name)
{
  if (name == "full_body_mode") {
    return Mode::FullBody;
  }
  if (name == "split_mode") {
    return Mode::Split;
  }
  if (name == "arm_sdk_mode") {
    return Mode::ArmSdk;
  }
  throw std::runtime_error(
          "unknown mode '" + name +
          "' (expected full_body_mode, split_mode or arm_sdk_mode)");
}

namespace
{

[[noreturn]] void fail(const std::string & source, const std::string & what)
{
  throw std::runtime_error("robot profile " + source + ": " + what);
}

void reject_unknown_keys(
  const YAML::Node & map, const std::set<std::string> & allowed,
  const std::string & source, const std::string & where)
{
  if (!map.IsMap()) {
    fail(source, where + "must be a map");
  }
  for (const auto & entry : map) {
    const std::string key = entry.first.as<std::string>();
    if (allowed.count(key) == 0) {
      fail(source, where + "unknown key '" + key + "'");
    }
  }
}

YAML::Node require(
  const YAML::Node & map, const std::string & key, const std::string & source,
  const std::string & where)
{
  const YAML::Node node = map[key];
  if (!node) {
    fail(source, where + "missing key '" + key + "'");
  }
  return node;
}

template<typename T>
T scalar_as(
  const YAML::Node & node, const std::string & key, const std::string & source,
  const std::string & where)
{
  try {
    return node.as<T>();
  } catch (const YAML::Exception &) {
    fail(source, where + "key '" + key + "' has an unusable value");
  }
}

template<typename T>
T require_scalar(
  const YAML::Node & map, const std::string & key, const std::string & source,
  const std::string & where)
{
  return scalar_as<T>(require(map, key, source, where), key, source, where);
}

uint8_t require_byte(
  const YAML::Node & map, const std::string & key, const std::string & source)
{
  const int value = require_scalar<int>(map, key, source, "");
  if (value < 0 || value > 255) {
    fail(source, "key '" + key + "' must be in 0..255");
  }
  return static_cast<uint8_t>(value);
}

std::pair<YAML::Node, YAML::Node> require_pair(
  const YAML::Node & map, const std::string & key, const std::string & source,
  const std::string & where)
{
  const YAML::Node node = require(map, key, source, where);
  if (!node.IsSequence() || node.size() != 2) {
    fail(source, where + "key '" + key + "' must be a list of two numbers");
  }
  return {node[0], node[1]};
}

SlotRange require_range(
  const YAML::Node & map, const std::string & key, const std::string & source)
{
  const auto ends = require_pair(map, key, source, "");
  return SlotRange{
    scalar_as<int>(ends.first, key, source, ""), scalar_as<int>(ends.second, key, source, "")};
}

JointLimit parse_joint(
  const YAML::Node & node, size_t index, const std::string & source)
{
  const std::string where = "joints[" + std::to_string(index) + "]: ";
  reject_unknown_keys(node, {"name", "slot", "position", "velocity", "effort"}, source, where);

  JointLimit joint;
  joint.name = require_scalar<std::string>(node, "name", source, where);
  const std::string named = "joint '" + joint.name + "': ";
  joint.slot = require_scalar<int>(node, "slot", source, named);
  const auto position = require_pair(node, "position", source, named);
  joint.q_low = scalar_as<double>(position.first, "position", source, named);
  joint.q_high = scalar_as<double>(position.second, "position", source, named);
  joint.dq_max = require_scalar<double>(node, "velocity", source, named);
  joint.tau_max = require_scalar<double>(node, "effort", source, named);

  if (joint.slot != static_cast<int>(index)) {
    fail(source, named + "slot " + std::to_string(joint.slot) +
           " is not its position " + std::to_string(index) + " in the joint list");
  }
  if (!std::isfinite(joint.q_low) || !std::isfinite(joint.q_high) ||
    !(joint.q_low < joint.q_high))
  {
    fail(source, named + "position must be finite [low, high] with low < high");
  }
  if (!std::isfinite(joint.dq_max) || !(joint.dq_max > 0.0)) {
    fail(source, named + "velocity must be finite and positive");
  }
  if (!std::isfinite(joint.tau_max) || !(joint.tau_max > 0.0)) {
    fail(source, named + "effort must be finite and positive");
  }
  return joint;
}

void check_range(
  const SlotRange & range, const std::string & key, size_t joint_count,
  const std::string & source)
{
  if (range.first < 0 || range.first > range.last ||
    range.last >= static_cast<int>(joint_count))
  {
    fail(source, "key '" + key + "' must be an inclusive range inside the " +
           std::to_string(joint_count) + " body joints");
  }
}

RobotProfile parse_checked(const YAML::Node & root, const std::string & source)
{
  reject_unknown_keys(
    root,
    {"robot", "mode_machine", "mode_pr", "motor_slots", "modes", "lower_slots",
      "upper_slots", "arm_sdk_weight_slot", "joints"},
    source, "");

  RobotProfile profile;
  profile.robot = require_scalar<std::string>(root, "robot", source, "");
  if (profile.robot.empty()) {
    fail(source, "key 'robot' is empty");
  }
  profile.mode_machine = require_byte(root, "mode_machine", source);
  profile.mode_pr = require_byte(root, "mode_pr", source);
  profile.motor_slots = require_scalar<int>(root, "motor_slots", source, "");

  const YAML::Node modes = require(root, "modes", source, "");
  if (!modes.IsSequence() || modes.size() == 0) {
    fail(source, "key 'modes' must be a non-empty list");
  }
  for (const auto & entry : modes) {
    const std::string name = scalar_as<std::string>(entry, "modes", source, "");
    try {
      profile.modes.push_back(parse_mode(name));
    } catch (const std::runtime_error & e) {
      fail(source, std::string("key 'modes': ") + e.what());
    }
  }

  const YAML::Node joints = require(root, "joints", source, "");
  if (!joints.IsSequence() || joints.size() == 0) {
    fail(source, "key 'joints' must be a non-empty list");
  }
  std::set<std::string> names;
  for (size_t i = 0; i < joints.size(); ++i) {
    profile.joints.push_back(parse_joint(joints[i], i, source));
    if (!names.insert(profile.joints.back().name).second) {
      fail(source, "joint '" + profile.joints.back().name + "' is listed twice");
    }
  }

  const size_t joint_count = profile.joints.size();
  if (profile.motor_slots < static_cast<int>(joint_count)) {
    fail(source, "key 'motor_slots' is smaller than the " + std::to_string(joint_count) +
           " body joints");
  }
  profile.lower_slots = require_range(root, "lower_slots", source);
  profile.upper_slots = require_range(root, "upper_slots", source);
  check_range(profile.lower_slots, "lower_slots", joint_count, source);
  check_range(profile.upper_slots, "upper_slots", joint_count, source);
  if (profile.lower_slots.first <= profile.upper_slots.last &&
    profile.upper_slots.first <= profile.lower_slots.last)
  {
    fail(source, "keys 'lower_slots' and 'upper_slots' overlap");
  }

  const bool arm_sdk = std::find(profile.modes.begin(), profile.modes.end(), Mode::ArmSdk) !=
    profile.modes.end();
  if (root["arm_sdk_weight_slot"]) {
    const int slot = require_scalar<int>(root, "arm_sdk_weight_slot", source, "");
    if (slot < static_cast<int>(joint_count) || slot >= profile.motor_slots) {
      fail(source, "key 'arm_sdk_weight_slot' must be a slot after the body joints and below "
           "motor_slots");
    }
    profile.arm_sdk_weight_slot = slot;
  }
  if (arm_sdk != profile.arm_sdk_weight_slot.has_value()) {
    fail(source, arm_sdk ?
           "modes lists arm_sdk_mode, which needs key 'arm_sdk_weight_slot'" :
           "key 'arm_sdk_weight_slot' is set but modes lacks arm_sdk_mode");
  }
  return profile;
}

}  // namespace

RobotProfile parse_robot_profile(const std::string & yaml_text, const std::string & source)
{
  YAML::Node root;
  try {
    root = YAML::Load(yaml_text);
  } catch (const YAML::Exception & e) {
    fail(source, e.what());
  }
  try {
    return parse_checked(root, source);
  } catch (const YAML::Exception & e) {
    fail(source, e.what());
  }
}

RobotProfile load_robot_profile(const std::string & robot)
{
  std::string path;
  try {
    path = ament_index_cpp::get_package_share_directory("golem_safety") + "/robots/" +
      robot + ".yaml";
  } catch (const std::exception & e) {
    throw std::runtime_error(
            "cannot load the profile of robot '" + robot + "': " + e.what());
  }
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("no profile for robot '" + robot + "': cannot read " + path);
  }
  std::ostringstream text;
  text << file.rdbuf();
  RobotProfile profile = parse_robot_profile(text.str(), path);
  if (profile.robot != robot) {
    fail(path, "key 'robot' is '" + profile.robot + "', expected '" + robot + "'");
  }
  return profile;
}

}  // namespace golem_safety
