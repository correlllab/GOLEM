#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "golem_safety/config.hpp"
#include "golem_safety/lowcmd.hpp"
#include "golem_safety/robot_profile.hpp"
#include "unitree_hg/msg/low_cmd.hpp"

using golem_safety::EstopMode;
using golem_safety::RobotProfile;
using golem_safety::SafetyConfig;
using unitree_hg::msg::LowCmd;

namespace {

const RobotProfile & h12()
{
  static const RobotProfile profile = golem_safety::load_robot_profile("h12");
  return profile;
}

const RobotProfile & g1()
{
  static const RobotProfile profile = golem_safety::load_robot_profile("g1");
  return profile;
}

// test/fixtures/lowcmd_crc.json, written by tools/crc_fixtures.py with the Unitree SDK.
// A command entry holds the message fields as columns (one list per field, 35 slots)
// and the CRC the SDK computed for it.
YAML::Node fixture()
{
  static const YAML::Node root =
    YAML::LoadFile(std::string(GOLEM_SAFETY_FIXTURE_DIR) + "/lowcmd_crc.json");
  return root;
}

LowCmd command_of(const YAML::Node & entry)
{
  LowCmd cmd;
  cmd.mode_pr = static_cast<uint8_t>(entry["mode_pr"].as<int>());
  cmd.mode_machine = static_cast<uint8_t>(entry["mode_machine"].as<int>());
  for (size_t i = 0; i < cmd.motor_cmd.size(); ++i) {
    cmd.motor_cmd[i].mode = static_cast<uint8_t>(entry["mode"][i].as<int>());
    cmd.motor_cmd[i].q = entry["q"][i].as<float>();
    cmd.motor_cmd[i].dq = entry["dq"][i].as<float>();
    cmd.motor_cmd[i].tau = entry["tau"][i].as<float>();
    cmd.motor_cmd[i].kp = entry["kp"][i].as<float>();
    cmd.motor_cmd[i].kd = entry["kd"][i].as<float>();
    cmd.motor_cmd[i].reserve = entry["motor_reserve"][i].as<uint32_t>();
  }
  for (size_t i = 0; i < cmd.reserve.size(); ++i) {
    cmd.reserve[i] = entry["reserve"][i].as<uint32_t>();
  }
  cmd.crc = entry["crc_field"].as<uint32_t>();
  return cmd;
}

// Every field equal, bit for bit.
void expect_same(const LowCmd & actual, const LowCmd & expected)
{
  EXPECT_EQ(actual.mode_pr, expected.mode_pr);
  EXPECT_EQ(actual.mode_machine, expected.mode_machine);
  for (size_t i = 0; i < expected.motor_cmd.size(); ++i) {
    SCOPED_TRACE("slot " + std::to_string(i));
    EXPECT_EQ(actual.motor_cmd[i].mode, expected.motor_cmd[i].mode);
    EXPECT_EQ(actual.motor_cmd[i].q, expected.motor_cmd[i].q);
    EXPECT_EQ(actual.motor_cmd[i].dq, expected.motor_cmd[i].dq);
    EXPECT_EQ(actual.motor_cmd[i].tau, expected.motor_cmd[i].tau);
    EXPECT_EQ(actual.motor_cmd[i].kp, expected.motor_cmd[i].kp);
    EXPECT_EQ(actual.motor_cmd[i].kd, expected.motor_cmd[i].kd);
    EXPECT_EQ(actual.motor_cmd[i].reserve, expected.motor_cmd[i].reserve);
  }
  EXPECT_EQ(actual.reserve, expected.reserve);
  EXPECT_EQ(actual.crc, expected.crc);
}

// Any LowCmd with distinct nonzero values, for tests that must not see them again.
LowCmd busy_command()
{
  LowCmd cmd;
  cmd.mode_pr = 1;
  cmd.mode_machine = 99;
  for (size_t i = 0; i < cmd.motor_cmd.size(); ++i) {
    cmd.motor_cmd[i].mode = 1;
    cmd.motor_cmd[i].q = 0.1F * static_cast<float>(i + 1);
    cmd.motor_cmd[i].dq = 0.2F;
    cmd.motor_cmd[i].tau = 3.0F;
    cmd.motor_cmd[i].kp = 100.0F;
    cmd.motor_cmd[i].kd = 5.0F;
    cmd.motor_cmd[i].reserve = 7;
  }
  cmd.reserve = {1, 2, 3, 4};
  cmd.crc = 0xABCDEF01U;
  return cmd;
}

}  // namespace

TEST(LowCmd, CrcMatchesSdkFixtures)
{
  const YAML::Node commands = fixture()["commands"];
  ASSERT_EQ(commands.size(), 5u);
  for (const YAML::Node & entry : commands) {
    SCOPED_TRACE(entry["name"].as<std::string>());
    EXPECT_EQ(golem_safety::lowcmd_crc(command_of(entry)), entry["crc"].as<uint32_t>());
  }
}

TEST(LowCmd, CrcCoversEveryFieldButTheCrc)
{
  const LowCmd base = busy_command();
  const uint32_t reference = golem_safety::lowcmd_crc(base);

  LowCmd other = base;
  other.crc = 0;
  EXPECT_EQ(golem_safety::lowcmd_crc(other), reference) << "the crc field is not covered";

  const auto changes = [&](const std::function<void(LowCmd &)> & edit) {
      LowCmd cmd = base;
      edit(cmd);
      return golem_safety::lowcmd_crc(cmd) != reference;
    };
  EXPECT_TRUE(changes([](LowCmd & c) {c.mode_pr ^= 1;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.mode_machine ^= 1;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[0].mode ^= 1;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[34].q += 1.0F;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[17].dq += 1.0F;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[17].tau += 1.0F;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[3].kp += 1.0F;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[3].kd += 1.0F;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.motor_cmd[34].reserve ^= 1U;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.reserve[0] ^= 1U;}));
  EXPECT_TRUE(changes([](LowCmd & c) {c.reserve[3] ^= 1U;}));
}

TEST(LowCmd, StampSetsProfileHeader)
{
  LowCmd cmd = busy_command();
  golem_safety::stamp_header_and_crc(cmd, h12());
  EXPECT_EQ(cmd.mode_machine, 6);
  EXPECT_EQ(cmd.mode_pr, 0);
  EXPECT_EQ(cmd.crc, golem_safety::lowcmd_crc(cmd));

  cmd = busy_command();
  golem_safety::stamp_header_and_crc(cmd, g1());
  EXPECT_EQ(cmd.mode_machine, 5);
  EXPECT_EQ(cmd.mode_pr, 0);
  EXPECT_EQ(cmd.crc, golem_safety::lowcmd_crc(cmd));

  // The CRC is over the stamped header: the same body, the other robot's header.
  LowCmd other = busy_command();
  golem_safety::stamp_header_and_crc(other, h12());
  EXPECT_NE(other.crc, cmd.crc);
}

TEST(LowCmd, ZeroCommandMatchesTodaysH12EstopCommand)
{
  // make_estop_cmd(LowCmdDefault()) of h12_safety_layer, with the header the new
  // layer stamps on every command; the fixture holds the SDK's CRC for it.
  const YAML::Node entry = fixture()["old_commands"]["h12_estop"];
  const LowCmd expected = command_of(entry);
  const LowCmd zero = golem_safety::zero_command(h12());
  expect_same(zero, expected);
  EXPECT_EQ(zero.crc, entry["crc"].as<uint32_t>());
}

TEST(LowCmd, ZeroCommandIsEveryMotorOff)
{
  for (const RobotProfile * profile : {&h12(), &g1()}) {
    SCOPED_TRACE(profile->robot);
    const LowCmd zero = golem_safety::zero_command(*profile);
    EXPECT_EQ(zero.mode_machine, profile->mode_machine);
    EXPECT_EQ(zero.mode_pr, profile->mode_pr);
    ASSERT_EQ(zero.motor_cmd.size(), static_cast<size_t>(profile->motor_slots));
    for (const auto & motor : zero.motor_cmd) {
      EXPECT_EQ(motor.mode, 0);
      EXPECT_EQ(motor.q, 0.0F);
      EXPECT_EQ(motor.dq, 0.0F);
      EXPECT_EQ(motor.tau, 0.0F);
      EXPECT_EQ(motor.kp, 0.0F);
      EXPECT_EQ(motor.kd, 0.0F);
      EXPECT_EQ(motor.reserve, 0u);
    }
    for (const uint32_t word : zero.reserve) {
      EXPECT_EQ(word, 0u);
    }
    EXPECT_EQ(zero.crc, golem_safety::lowcmd_crc(zero));
  }
}

TEST(LowCmd, EstopCommandZeroIsTheZeroCommand)
{
  SafetyConfig config = golem_safety::load_safety_config("split", g1());
  config.estop_response.mode = EstopMode::Zero;
  const LowCmd last = busy_command();
  expect_same(golem_safety::estop_command(g1(), config, last), golem_safety::zero_command(g1()));

  SafetyConfig h12_config = golem_safety::load_safety_config("default_safety_full", h12());
  EXPECT_EQ(h12_config.estop_response.mode, EstopMode::Zero);
  expect_same(
    golem_safety::estop_command(h12(), h12_config, last), golem_safety::zero_command(h12()));
}

TEST(LowCmd, DampKeepsKd)
{
  SafetyConfig config = golem_safety::load_safety_config("split", g1());
  config.estop_response.mode = EstopMode::Damp;
  const size_t body = g1().joints.size();
  config.estop_response.kd.assign(body, 0.0);
  for (size_t i = 0; i < body; ++i) {
    config.estop_response.kd[i] = 0.5 * static_cast<double>(1 + i % 16);
  }

  const LowCmd last = busy_command();
  const LowCmd cmd = golem_safety::estop_command(g1(), config, last);
  const LowCmd zero = golem_safety::zero_command(g1());

  EXPECT_EQ(cmd.mode_machine, 5);
  EXPECT_EQ(cmd.mode_pr, 0);
  for (size_t i = 0; i < body; ++i) {
    SCOPED_TRACE("body slot " + std::to_string(i));
    EXPECT_EQ(cmd.motor_cmd[i].mode, 1);
    EXPECT_EQ(cmd.motor_cmd[i].q, 0.0F);
    EXPECT_EQ(cmd.motor_cmd[i].dq, 0.0F);
    EXPECT_EQ(cmd.motor_cmd[i].tau, 0.0F);
    EXPECT_EQ(cmd.motor_cmd[i].kp, 0.0F);
    EXPECT_EQ(cmd.motor_cmd[i].kd, static_cast<float>(config.estop_response.kd[i]));
  }
  // Slots past the body, among them the arm_sdk weight in slot 29, stay as the zero
  // command has them: nothing of `last` reaches the result.
  for (size_t i = body; i < cmd.motor_cmd.size(); ++i) {
    SCOPED_TRACE("slot " + std::to_string(i));
    EXPECT_EQ(cmd.motor_cmd[i].mode, zero.motor_cmd[i].mode);
    EXPECT_EQ(cmd.motor_cmd[i].q, zero.motor_cmd[i].q);
    EXPECT_EQ(cmd.motor_cmd[i].dq, zero.motor_cmd[i].dq);
    EXPECT_EQ(cmd.motor_cmd[i].tau, zero.motor_cmd[i].tau);
    EXPECT_EQ(cmd.motor_cmd[i].kp, zero.motor_cmd[i].kp);
    EXPECT_EQ(cmd.motor_cmd[i].kd, zero.motor_cmd[i].kd);
    EXPECT_EQ(cmd.motor_cmd[i].reserve, zero.motor_cmd[i].reserve);
  }
  EXPECT_EQ(cmd.reserve, zero.reserve);
  EXPECT_EQ(cmd.crc, golem_safety::lowcmd_crc(cmd));
}

TEST(LowCmd, DampMatchesTodaysG1EstopCommand)
{
  // make_estop_cmd(template, kd) of g1_safety_layer; the fixture holds its kd and the
  // SDK's CRC for the result.
  const YAML::Node entry = fixture()["old_commands"]["g1_damp"];
  SafetyConfig config = golem_safety::load_safety_config("split", g1());
  config.estop_response.mode = EstopMode::Damp;
  config.estop_response.kd.clear();
  for (const YAML::Node & kd : entry["kd_input"]) {
    config.estop_response.kd.push_back(kd.as<double>());
  }
  ASSERT_EQ(config.estop_response.kd.size(), g1().joints.size());

  const LowCmd cmd = golem_safety::estop_command(g1(), config, busy_command());
  expect_same(cmd, command_of(entry));
  EXPECT_EQ(cmd.crc, entry["crc"].as<uint32_t>());
}

TEST(LowCmd, DampNeedsAKdPerBodySlot)
{
  SafetyConfig config = golem_safety::load_safety_config("split", g1());
  config.estop_response.mode = EstopMode::Damp;
  config.estop_response.kd.assign(g1().joints.size() - 1, 1.0);
  EXPECT_THROW(golem_safety::estop_command(g1(), config, busy_command()), std::invalid_argument);
}
