// golem_safety_dump_config <robot> <config>
//
// Loads a safety config with the node's own loader and prints what it derived as
// JSON: the derived limits and every parsed value, under the schema's keys. The
// limit parity test compares this output to what the old safety layers derive.
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"

namespace
{

using golem_safety::EstopAfter;
using golem_safety::EstopMode;
using golem_safety::StartupRule;

// A double that reads back as the same double.
std::string number(double value)
{
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return out.str();
}

std::string quoted(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out + "\"";
}

std::string array(const std::vector<double> & values)
{
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    out += (i ? ", " : "") + number(values[i]);
  }
  return out + "]";
}

std::string boolean(bool value) {return value ? "true" : "false";}

void print(const golem_safety::SafetyConfig & c, const golem_safety::RobotProfile & profile)
{
  const auto & l = c.limits;
  std::cout << "{\n"
            << " \"robot\": " << quoted(c.robot) << ",\n"
            << " \"source\": " << quoted(c.source) << ",\n"
            << " \"mode_machine\": " << static_cast<int>(profile.mode_machine) << ",\n"
            << " \"mode\": " << quoted(golem_safety::mode_name(c.mode)) << ",\n"
            << " \"topics\": {\"low_cmd_in\": " << quoted(c.topics.low_cmd_in)
            << ", \"low_cmd_lower_in\": " << quoted(c.topics.low_cmd_lower_in)
            << ", \"low_cmd_upper_in\": " << quoted(c.topics.low_cmd_upper_in)
            << ", \"low_cmd_out\": " << quoted(c.topics.low_cmd_out)
            << ", \"low_state\": " << quoted(c.topics.low_state) << "},\n"
            << " \"control\": {\"publish_hz\": " << number(c.control.publish_hz)
            << ", \"state_lag_ms\": " << number(c.control.state_lag_ms) << "},\n"
            << " \"stale\": {\"upper_seconds\": " << number(c.stale.upper_seconds)
            << ", \"lower_seconds\": " << number(c.stale.lower_seconds)
            << ", \"state_seconds\": " << number(c.stale.state_seconds) << "},\n"
            << " \"single_writer\": {\"handover_seconds\": "
            << number(c.single_writer.handover_seconds) << "},\n"
            << " \"startup\": {\"rule\": "
            << quoted(c.startup.rule == StartupRule::ZeroStream ? "zero_stream" : "engage_gate")
            << "},\n"
            << " \"engage\": {\"max_state_age_seconds\": "
            << number(c.engage.max_state_age_seconds) << ", \"max_position_error\": "
            << number(c.engage.max_position_error) << "},\n"
            << " \"estop\": {\"triggered_topic\": " << quoted(c.estop.triggered_topic)
            << ", \"plugged_in_topic\": " << quoted(c.estop.plugged_in_topic)
            << ", \"timeout_seconds\": " << number(c.estop.timeout_seconds) << "},\n"
            << " \"estop_response\": {\"mode\": "
            << quoted(c.estop_response.mode == EstopMode::Zero ? "zero" : "damp")
            << ", \"kd\": " << array(c.estop_response.kd) << ", \"after\": "
            << quoted(c.estop_response.after == EstopAfter::BurstThenSilence ?
      "burst_then_silence" : "keep_writing")
            << ", \"burst_count\": " << c.estop_response.burst_count << "},\n"
            << " \"limits\": {\n"
            << "  \"q_clip_low\": " << array(l.q_clip_low) << ",\n"
            << "  \"q_clip_high\": " << array(l.q_clip_high) << ",\n"
            << "  \"dq_clip\": " << array(l.dq_clip) << ",\n"
            << "  \"tau_clip\": " << array(l.tau_clip) << ",\n"
            << "  \"q_estop_low\": " << array(l.q_estop_low) << ",\n"
            << "  \"q_estop_high\": " << array(l.q_estop_high) << ",\n"
            << "  \"dq_estop\": " << array(l.dq_estop) << ",\n"
            << "  \"tau_estop\": " << array(l.tau_estop) << ",\n"
            << "  \"kp_max\": " << number(l.kp_max) << ",\n"
            << "  \"kd_max\": " << number(l.kd_max) << "\n"
            << " },\n"
            << " \"logging\": {\"enabled\": " << boolean(c.logging.enabled)
            << ", \"base_dir\": " << quoted(c.logging.base_dir)
            << ", \"chunk_prefix\": " << quoted(c.logging.chunk_prefix) << "}\n"
            << "}\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <robot> <config name or path>\n", argv[0]);
    return 2;
  }
  try {
    const golem_safety::RobotProfile profile = golem_safety::load_robot_profile(argv[1]);
    print(golem_safety::load_safety_config(argv[2], profile), profile);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
