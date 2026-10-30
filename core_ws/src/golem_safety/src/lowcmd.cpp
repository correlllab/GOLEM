// The CRC is adapted from unitree_ros2's example/src/src/common/motor_crc_hg.cpp and
// include/common/motor_crc_hg.h, under the licence below.
//
// Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
// Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
// All rights reserved.
//
// BSD 3-Clause License
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "golem_safety/lowcmd.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace golem_safety
{

namespace
{

constexpr size_t kSlots = 35;

// The LowCmd struct of motor_crc_hg.h, which is the wire layout the robot checksums.
struct RawMotorCmd
{
  uint8_t mode;
  float q;
  float dq;
  float tau;
  float kp;
  float kd;
  uint32_t reserve;
};

struct RawLowCmd
{
  uint8_t mode_pr;
  uint8_t mode_machine;
  std::array<RawMotorCmd, kSlots> motor_cmd;
  std::array<uint32_t, 4> reserve;
  uint32_t crc;
};

static_assert(std::is_trivially_copyable<RawLowCmd>::value, "memcpy and memset need this");
static_assert(sizeof(RawMotorCmd) == 28, "motor_crc_hg.h MotorCmd is 28 bytes");
static_assert(sizeof(RawLowCmd) == 1004, "motor_crc_hg.h LowCmd is 1004 bytes");
static_assert(
  std::tuple_size<decltype(unitree_hg::msg::LowCmd::motor_cmd)>::value == kSlots,
  "the CRC covers 35 motor slots");

constexpr size_t kCrcWords = sizeof(RawLowCmd) / 4 - 1;  // every word but crc

// Over the first `len` words of `ptr`, bit by bit, MSB first.
uint32_t crc32_core(const uint32_t * ptr, size_t len)
{
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (size_t i = 0; i < len; i++) {
    uint32_t xbit = 1U << 31;
    const uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) {
        CRC32 ^= dwPolynomial;
      }
      xbit >>= 1;
    }
  }
  return CRC32;
}

}  // namespace

uint32_t lowcmd_crc(const unitree_hg::msg::LowCmd & cmd)
{
  // The padding after each uint8_t is part of the checksummed words, so it is zeroed.
  RawLowCmd raw;
  std::memset(&raw, 0, sizeof(raw));
  raw.mode_pr = cmd.mode_pr;
  raw.mode_machine = cmd.mode_machine;
  for (size_t i = 0; i < kSlots; ++i) {
    const auto & motor = cmd.motor_cmd[i];
    raw.motor_cmd[i].mode = motor.mode;
    raw.motor_cmd[i].q = motor.q;
    raw.motor_cmd[i].dq = motor.dq;
    raw.motor_cmd[i].tau = motor.tau;
    raw.motor_cmd[i].kp = motor.kp;
    raw.motor_cmd[i].kd = motor.kd;
    raw.motor_cmd[i].reserve = motor.reserve;
  }
  // All four reserve words are covered, as in the Unitree SDK's CRC (the example's get_crc
  // copies reserve[0] only).
  raw.reserve = cmd.reserve;

  std::array<uint32_t, sizeof(RawLowCmd) / 4> words;
  std::memcpy(words.data(), &raw, sizeof(raw));
  return crc32_core(words.data(), kCrcWords);
}

void stamp_header_and_crc(unitree_hg::msg::LowCmd & cmd, const RobotProfile & profile)
{
  cmd.mode_pr = profile.mode_pr;
  cmd.mode_machine = profile.mode_machine;
  cmd.crc = lowcmd_crc(cmd);
}

unitree_hg::msg::LowCmd zero_command(const RobotProfile & profile)
{
  unitree_hg::msg::LowCmd cmd;
  for (auto & motor : cmd.motor_cmd) {
    motor.mode = 0;
    motor.q = 0.0F;
    motor.dq = 0.0F;
    motor.tau = 0.0F;
    motor.kp = 0.0F;
    motor.kd = 0.0F;
    motor.reserve = 0;
  }
  cmd.reserve.fill(0);
  stamp_header_and_crc(cmd, profile);
  return cmd;
}

unitree_hg::msg::LowCmd estop_command(
  const RobotProfile & profile, const SafetyConfig & config,
  const unitree_hg::msg::LowCmd & /* last */)
{
  unitree_hg::msg::LowCmd cmd = zero_command(profile);
  if (config.estop_response.mode == EstopMode::Zero) {
    return cmd;
  }
  const std::vector<double> & kd = config.estop_response.kd;
  if (kd.size() != profile.joints.size()) {
    throw std::invalid_argument(
            "estop_response.kd has " + std::to_string(kd.size()) + " entries for the " +
            profile.robot + "'s " + std::to_string(profile.joints.size()) + " body slots");
  }
  for (size_t slot = 0; slot < kd.size(); ++slot) {
    cmd.motor_cmd[slot].mode = 1;
    cmd.motor_cmd[slot].kd = static_cast<float>(kd[slot]);
  }
  stamp_header_and_crc(cmd, profile);
  return cmd;
}

}  // namespace golem_safety
