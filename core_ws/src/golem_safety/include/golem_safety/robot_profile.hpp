#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace golem_safety
{

// One body joint. The slot is its index in the LowCmd/LowState motor arrays.
struct JointLimit
{
  std::string name;
  int slot = 0;
  double q_low = 0.0;    // rad
  double q_high = 0.0;   // rad
  double dq_max = 0.0;   // rad/s
  double tau_max = 0.0;  // N*m
};

// An inclusive range of motor slots.
struct SlotRange
{
  int first = 0;
  int last = 0;
  bool contains(int slot) const {return slot >= first && slot <= last;}
};

enum class Mode { FullBody, Split, ArmSdk };

// Accepts full_body_mode, split_mode and arm_sdk_mode; throws std::runtime_error
// for anything else.
Mode parse_mode(const std::string & name);

// The name parse_mode accepts for `mode`.
const char * mode_name(Mode mode);

// Everything the safety layer knows about one robot. The data file is
// share/golem_safety/robots/<robot>.yaml, written by tools/generate_robot_data.py
// and read by golem_upper_ik too.
struct RobotProfile
{
  std::string robot;
  uint8_t mode_machine = 0;  // what /lowstate reports and every command carries
  uint8_t mode_pr = 0;
  int motor_slots = 0;       // length of the LowCmd/LowState motor arrays
  std::vector<JointLimit> joints;  // indexed by slot
  std::vector<Mode> modes;
  SlotRange lower_slots, upper_slots;
  std::optional<int> arm_sdk_weight_slot;  // q of this slot is the arm_sdk blend weight
};

// Parses and validates profile text. `source` names the text in error messages.
// Throws std::runtime_error on a missing or unknown key, a malformed value or
// an inconsistent profile.
RobotProfile parse_robot_profile(const std::string & yaml_text, const std::string & source);

// Loads share/golem_safety/robots/<robot>.yaml. Throws std::runtime_error naming
// the robot when the file is missing or invalid.
RobotProfile load_robot_profile(const std::string & robot);

}  // namespace golem_safety
