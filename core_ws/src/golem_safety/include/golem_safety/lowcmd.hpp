#pragma once

#include <cstdint>

#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"
#include "unitree_hg/msg/low_cmd.hpp"

namespace golem_safety
{

// Return the Unitree LowCmd CRC: CRC-32 (polynomial 0x04c11db7, initial value 0xffffffff,
// no reflection, no final xor) over the message's little-endian 32-bit words as the C struct
// of the Unitree SDK lays them out, padding bytes zero, every word but the crc field itself.
// The robots and the sims drop a command whose CRC does not match.
uint32_t lowcmd_crc(const unitree_hg::msg::LowCmd & cmd);

// Set mode_pr and mode_machine from `profile`, then crc.
// SAFETY: every command is stamped last, after its final edit; an edit after the stamp
// needs a new stamp, or the robot drops the command.
void stamp_header_and_crc(unitree_hg::msg::LowCmd & cmd, const RobotProfile & profile);

// Return the stamped command that turns every motor off: mode 0 and q, dq, tau, kp, kd and
// every reserve word 0 on every slot.
unitree_hg::msg::LowCmd zero_command(const RobotProfile & profile);

// Return the stamped command written after an estop. Zero is zero_command(); Damp starts
// from it and sets every body slot to mode 1, kp 0 and the config's kd, so the robot sinks
// on its dampers instead of dropping. Throws std::invalid_argument when Damp's kd has not
// one entry per body slot.
// SAFETY: it depends on the profile and the config only, never on a controller's command.
unitree_hg::msg::LowCmd estop_command(const RobotProfile & profile, const SafetyConfig & config);

}  // namespace golem_safety
