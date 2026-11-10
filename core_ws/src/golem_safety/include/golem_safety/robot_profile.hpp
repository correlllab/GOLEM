#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace golem_safety
{

// Hold one body joint's limits from the robot profile.
struct JointLimit
{
  std::string name;
  int slot = 0;          // its index in the LowCmd/LowState motor arrays
  double q_low = 0.0;    // rad
  double q_high = 0.0;   // rad
  double dq_max = 0.0;   // rad/s
  double tau_max = 0.0;  // N*m
};

// Hold an inclusive range of motor slots.
struct SlotRange
{
  int first = 0;
  int last = 0;
  // Return whether `slot` lies in the range.
  bool contains(int slot) const {return slot >= first && slot <= last;}
};

// The length of unitree_hg's LowCmd/LowState motor arrays, which every profile must state.
constexpr int kMotorSlots = 35;

// Name how the layer reads its inputs: one full-body input, or a lower and an upper one.
enum class Mode { FullBody, Split };

// Return the Mode named `name` (full_body_mode or split_mode); throw std::runtime_error for
// anything else.
Mode parse_mode(const std::string & name);

// Return the name parse_mode() accepts for `mode`.
const char * mode_name(Mode mode);

// Hold everything the safety layer knows about one robot, read from
// share/golem_safety/robots/<robot>.yaml (written by tools/generate_robot_data.py and read by
// golem_upper_ik too).
struct RobotProfile
{
  std::string robot;
  uint8_t mode_machine = 0;  // what /lowstate reports and every command carries
  uint8_t mode_pr = 0;
  int motor_slots = 0;       // kMotorSlots
  std::vector<JointLimit> joints;  // the body joints, indexed by slot
  std::vector<Mode> modes;   // each once; a config without a mode runs the first
  SlotRange lower_slots, upper_slots;  // together, every body joint once
};

// Parse and validate profile text; `source` names the text in errors. Throws
// std::runtime_error on a missing or unknown key, a malformed value, or an inconsistent
// profile: motor_slots other than kMotorSlots, more body joints than that, a mode listed
// twice, or lower and upper slot ranges that do not cover every body joint once.
RobotProfile parse_robot_profile(const std::string & yaml_text, const std::string & source);

// Load share/golem_safety/robots/<robot>.yaml. Throws std::runtime_error naming the robot
// when the file is missing or invalid, or names another robot.
RobotProfile load_robot_profile(const std::string & robot);

}  // namespace golem_safety
