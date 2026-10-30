#pragma once

#include <cstdint>

#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"
#include "unitree_hg/msg/low_cmd.hpp"

namespace golem_safety
{

// The Unitree LowCmd CRC: CRC-32 (polynomial 0x04c11db7, initial value 0xffffffff, no
// reflection, no final xor) over the message's little-endian 32-bit words, every word but
// the crc field itself. The words are the C struct's, with its padding bytes zero.
uint32_t lowcmd_crc(const unitree_hg::msg::LowCmd & cmd);

// Sets mode_pr and mode_machine from `profile`, then crc. A command is published only
// after this, and an edit after it needs a new stamp.
void stamp_header_and_crc(unitree_hg::msg::LowCmd & cmd, const RobotProfile & profile);

// Every motor off: mode 0 and q, dq, tau, kp and kd all 0 on every slot. Stamped.
unitree_hg::msg::LowCmd zero_command(const RobotProfile & profile);

// The command written after an estop, stamped. Zero is zero_command; Damp starts from it
// and sets every body slot to mode 1, kp 0 and the config's kd (one entry per body slot),
// so the slots past the body stay as zero_command has them. The result depends on the
// profile and the config only: nothing of `last`, the newest command of any input, is
// copied. Throws std::invalid_argument when Damp's kd has not one entry per body slot.
unitree_hg::msg::LowCmd estop_command(
  const RobotProfile & profile, const SafetyConfig & config,
  const unitree_hg::msg::LowCmd & last);

}  // namespace golem_safety
