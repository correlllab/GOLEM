#pragma once

#include <string>
#include <vector>

#include "golem_safety/robot_profile.hpp"

namespace golem_safety
{

// Per-slot limits derived from the robot's joint table and the config's ratios,
// indexed by body slot (one entry per body joint).
//
//   q_*_low  = joint low  + position_offset      rad
//   q_*_high = joint high - position_offset      rad
//   dq_*     = joint velocity limit * velocity_ratio   rad/s
//   tau_*    = joint effort limit   * torque_ratio     N*m
//
// The clip limits bound what a controller may command; the estop limits bound
// what the measured state may reach before the estop latches. kp_max and kd_max
// bound the commanded gains of every joint.
struct DerivedLimits
{
  std::vector<double> q_clip_low, q_clip_high, dq_clip, tau_clip;
  std::vector<double> q_estop_low, q_estop_high, dq_estop, tau_estop;
  double kp_max = 300.0;
  double kd_max = 30.0;
};

// What the layer does from boot until its first command is relayed. Every config
// names its rule (startup.rule has no default).
//   ZeroStream  streams a zero-gain command on every slot from boot (H1-2).
//   EngageGate  writes nothing until a fresh /lowstate and a command near the
//               measured joints pass the engage gate (G1).
enum class StartupRule { ZeroStream, EngageGate };

// What the layer does after the estop latches. Every config names it
// (estop_response.after has no default), and arm_sdk_mode requires KeepWriting.
//   BurstThenSilence  estop_response.burst_count commands, then it stops publishing.
//   KeepWriting       the estop command keeps being written.
enum class EstopAfter { BurstThenSilence, KeepWriting };

// The estop command: Zero turns every motor off; Damp keeps every motor on with
// kp = 0 and the configured kd.
enum class EstopMode { Zero, Damp };

struct Topics  // ROS topic names; the DDS topic rt/x is the ROS topic /x
{
  std::string low_cmd_in;        // full_body_mode input
  std::string low_cmd_lower_in;  // split_mode legs input
  std::string low_cmd_upper_in;  // split_mode and arm_sdk_mode waist + arms input
  std::string low_cmd_out;       // /arm_sdk by default in arm_sdk_mode, else /lowcmd
  std::string low_state;
};

struct Control
{
  double publish_hz = 500.0;
  // Estop when /lowstate arrives this many ms later than the robot's tick clock
  // implies; <= 0 disables it.
  double state_lag_ms = 100.0;
};

// Quiet-input thresholds, seconds. A quiet input or /lowstate is reported and its
// last command keeps being relayed; a value <= 0 disables the report.
struct Stale
{
  double upper_seconds = 0.0;
  double lower_seconds = 0.2;
  double state_seconds = 0.1;
};

struct SingleWriter
{
  // A new writer that appears after the old one was quiet longer than this is accepted;
  // two writers live at once latch the estop. Positive: at 0 any interleaving writer
  // would count as a handover.
  double handover_seconds = 0.5;
};

struct Startup
{
  StartupRule rule = StartupRule::ZeroStream;
};

// EngageGate only.
struct Engage
{
  double max_state_age_seconds = 0.1;
  double max_position_error = 0.15;  // rad
};

// arm_sdk_mode only.
struct ArmSdk
{
  double engage_seconds = 1.0;   // weight ramp 0 -> 1
  double release_seconds = 3.0;  // weight ramp 1 -> 0
  double waist_kp_min = 40.0;    // a command with a waist kp below this is rejected
};

// The hardware estop's ROS topics, read only when the bringup passes use_estop.
struct EstopInput
{
  std::string triggered_topic = "/estop/triggered";
  std::string plugged_in_topic = "/estop/plugged_in";
  double timeout_seconds = 0.5;  // a sample older than this latches
};

struct EstopResponse
{
  EstopMode mode = EstopMode::Zero;
  std::vector<double> kd;  // per body slot; used by Damp only
  EstopAfter after = EstopAfter::BurstThenSilence;
  int burst_count = 50;  // BurstThenSilence only
};

struct Logging
{
  bool enabled = false;
  std::string base_dir = "/tmp/golem_safety";
  std::string chunk_prefix = "chunk_";
};

// One parsed and validated safety config: one field per schema key.
struct SafetyConfig
{
  std::string robot;   // the profile's robot
  std::string source;  // the file or text name it was read from
  Mode mode = Mode::FullBody;
  Topics topics;
  Control control;
  Stale stale;
  SingleWriter single_writer;
  Startup startup;
  Engage engage;
  ArmSdk arm_sdk;
  EstopInput estop;
  EstopResponse estop_response;
  DerivedLimits limits;
  Logging logging;
};

// Parses and validates config text for `profile`. `source` names the text in error
// messages. Throws std::runtime_error, naming the robot, the source and the key, on
// an unknown key, a value out of range, a mode the profile lacks, or a per-joint
// list whose length is not the profile's joint count.
SafetyConfig parse_safety_config(
  const std::string & yaml_text, const std::string & source, const RobotProfile & profile);

// Loads a config for `profile`. A name without a directory, with or without ".yaml",
// is read from share/golem_safety/config/<profile.robot>/; anything else is a path.
// Throws std::runtime_error as parse_safety_config does, and when the file is missing.
SafetyConfig load_safety_config(const std::string & name_or_path, const RobotProfile & profile);

}  // namespace golem_safety
