#pragma once

#include <string>
#include <vector>

#include "golem_safety/robot_profile.hpp"

namespace golem_safety
{

// Hold the per-slot limits derived from the robot's joint table and the config's ratios,
// one entry per body joint, in slot order:
//
//   q_*_low  = joint low  + position_offset            rad
//   q_*_high = joint high - position_offset            rad
//   dq_*     = joint velocity limit * velocity_ratio   rad/s
//   tau_*    = joint effort limit   * torque_ratio     N*m
//
// SAFETY: the clip limits bound what a controller may command; the estop limits bound what
// the measured state may reach before the estop latches. kp_max and kd_max bound every
// joint's commanded gains. A negative position_offset widens a range past the joint table.
struct DerivedLimits
{
  std::vector<double> q_clip_low, q_clip_high, dq_clip, tau_clip;
  std::vector<double> q_estop_low, q_estop_high, dq_estop, tau_estop;
  double kp_max = 300.0;
  double kd_max = 30.0;
};

// Name what the layer writes after the estop latches (estop_response.after, which every
// config must give).
//   BurstThenSilence  estop_response.burst_count estop commands, then nothing.
//   KeepWriting       the estop command on every cycle until restart.
enum class EstopAfter { BurstThenSilence, KeepWriting };

// Name the estop command (estop_response.mode): Zero turns every motor off; Damp keeps
// every motor on with kp 0 and the configured kd.
enum class EstopMode { Zero, Damp };

// Hold the ROS topic names (topics.*); the DDS topic rt/x is the ROS topic /x.
struct Topics
{
  std::string low_cmd_in;        // full_body_mode input
  std::string low_cmd_lower_in;  // split_mode lower-body input
  std::string low_cmd_upper_in;  // split_mode upper-body input
  std::string low_cmd_out;       // the body command topic the node writes
  std::string low_state;
};

// Hold the relay loop's settings (control.*).
struct Control
{
  double publish_hz = 500.0;
  // ms; <= 0 turns the transport-lag check off. It has no default, so that no robot runs
  // another robot's value by leaving it out.
  double state_lag_ms = 0.0;
};

// Hold the quiet-report thresholds (stale.*), s; <= 0 turns a report off.
struct Stale
{
  double upper_seconds = 0.0;
  double lower_seconds = 0.2;
  double state_seconds = 0.1;
};

// Hold the single-writer rule's setting (single_writer.*).
struct SingleWriter
{
  // s a writer must be quiet before another writer takes its input over. Must be positive:
  // at 0 any two interleaving writers would count as a handover.
  double handover_seconds = 0.5;
};

// Hold the hardware estop's ROS topics and timeout (estop.*), read only with use_estop.
struct EstopInput
{
  std::string triggered_topic = "/estop/triggered";
  std::string plugged_in_topic = "/estop/plugged_in";
  double timeout_seconds = 0.5;  // s; a newer sample must arrive within this, or it latches
};

// Hold what the layer writes after the latch (estop_response.*).
struct EstopResponse
{
  EstopMode mode = EstopMode::Zero;
  std::vector<double> kd;  // per body slot; Damp only
  EstopAfter after = EstopAfter::BurstThenSilence;
  int burst_count = 50;  // BurstThenSilence only
};

// Hold the chunk logger's settings (logging.*).
struct Logging
{
  bool enabled = false;
  std::string base_dir = "/tmp/golem_safety";  // each run logs to a directory under it
  std::string chunk_prefix = "chunk_";
};

// Hold one parsed and validated safety config, one field per schema section.
struct SafetyConfig
{
  std::string robot;   // the profile's robot
  std::string source;  // the file or text name it was read from
  Mode mode = Mode::FullBody;
  Topics topics;
  Control control;
  Stale stale;
  SingleWriter single_writer;
  EstopInput estop;
  EstopResponse estop_response;
  DerivedLimits limits;
  Logging logging;
};

// Parse and validate config text for `profile`; `source` names the text in errors. Throws
// std::runtime_error, naming the robot, the source and the key, on an unknown key, a value
// out of range, a mode the profile lacks, or a per-joint list whose length is not the
// profile's joint count.
SafetyConfig parse_safety_config(
  const std::string & yaml_text, const std::string & source, const RobotProfile & profile);

// Load a config for `profile`. A name without a '/', with or without ".yaml", is read from
// share/golem_safety/config/<profile.robot>/; anything else is a path. Throws
// std::runtime_error as parse_safety_config() does, when the file cannot be read, and when
// the path's directory is named after another robot with a profile (config/g1/... for the
// h12), since several configs of one robot also parse for the other.
SafetyConfig load_safety_config(const std::string & name_or_path, const RobotProfile & profile);

}  // namespace golem_safety
