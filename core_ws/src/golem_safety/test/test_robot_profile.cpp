#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "golem_safety/robot_profile.hpp"

using golem_safety::JointLimit;
using golem_safety::Mode;
using golem_safety::RobotProfile;
using golem_safety::SlotRange;

namespace {

// The message of the std::runtime_error that `fn` throws; fails the test when
// it throws nothing.
std::string error_of(const std::function<void()> & fn)
{
  try {
    fn();
  } catch (const std::runtime_error & e) {
    return e.what();
  }
  ADD_FAILURE() << "expected std::runtime_error";
  return "";
}

// Two body joints, one lower and one upper slot.
const char * const kValid = R"(
robot: toy
mode_machine: 7
mode_pr: 0
motor_slots: 35
modes: [split_mode]
lower_slots: [0, 0]
upper_slots: [1, 1]
joints:
  - {name: a_joint, slot: 0, position: [-1.0, 1.0], velocity: 10.0, effort: 20.0}
  - {name: b_joint, slot: 1, position: [-2.0, 2.0], velocity: 11.0, effort: 21.0}
)";

std::string replaced(std::string text, const std::string & from, const std::string & to)
{
  const auto at = text.find(from);
  EXPECT_NE(at, std::string::npos) << "test text lacks: " << from;
  if (at != std::string::npos) {
    text.replace(at, from.size(), to);
  }
  return text;
}

std::string parse_error(const std::string & text)
{
  return error_of([&] {golem_safety::parse_robot_profile(text, "toy.yaml");});
}

}  // namespace

TEST(RobotProfile, LoadsH12)
{
  const RobotProfile p = golem_safety::load_robot_profile("h12");
  EXPECT_EQ(p.robot, "h12");
  EXPECT_EQ(p.mode_machine, 6);
  EXPECT_EQ(p.mode_pr, 0);
  EXPECT_EQ(p.motor_slots, 35);
  ASSERT_EQ(p.joints.size(), 27u);
  EXPECT_EQ(p.lower_slots.first, 0);
  EXPECT_EQ(p.lower_slots.last, 11);
  EXPECT_EQ(p.upper_slots.first, 12);
  EXPECT_EQ(p.upper_slots.last, 26);
  EXPECT_EQ(p.modes, (std::vector<Mode>{Mode::FullBody, Mode::Split}));

  for (size_t i = 0; i < p.joints.size(); ++i) {
    EXPECT_EQ(p.joints[i].slot, static_cast<int>(i));
  }
  EXPECT_EQ(p.joints[0].name, "left_hip_yaw_joint");
  EXPECT_EQ(p.joints[12].name, "torso_joint");
  EXPECT_EQ(p.joints[26].name, "right_wrist_yaw_joint");
}

TEST(RobotProfile, H12ShoulderRollKeepsTheTableLimitsNotTheUrdfs)
{
  const RobotProfile p = golem_safety::load_robot_profile("h12");
  const JointLimit & left = p.joints[14];
  EXPECT_EQ(left.name, "left_shoulder_roll_joint");
  EXPECT_DOUBLE_EQ(left.q_low, -0.38);
  EXPECT_DOUBLE_EQ(left.q_high, 3.4);
  EXPECT_DOUBLE_EQ(left.dq_max, 9.0);
  EXPECT_DOUBLE_EQ(left.tau_max, 40.0);
  const JointLimit & right = p.joints[21];
  EXPECT_EQ(right.name, "right_shoulder_roll_joint");
  EXPECT_DOUBLE_EQ(right.q_low, -3.4);
  EXPECT_DOUBLE_EQ(right.q_high, 0.38);
}

TEST(RobotProfile, LoadsG1)
{
  const RobotProfile p = golem_safety::load_robot_profile("g1");
  EXPECT_EQ(p.robot, "g1");
  EXPECT_EQ(p.mode_machine, 5);
  EXPECT_EQ(p.mode_pr, 0);
  EXPECT_EQ(p.motor_slots, 35);
  ASSERT_EQ(p.joints.size(), 29u);
  EXPECT_EQ(p.lower_slots.first, 0);
  EXPECT_EQ(p.lower_slots.last, 11);
  EXPECT_EQ(p.upper_slots.first, 12);
  EXPECT_EQ(p.upper_slots.last, 28);
  EXPECT_EQ(p.modes, (std::vector<Mode>{Mode::Split, Mode::FullBody}));

  for (size_t i = 0; i < p.joints.size(); ++i) {
    EXPECT_EQ(p.joints[i].slot, static_cast<int>(i));
  }
  const JointLimit & first = p.joints.front();
  EXPECT_EQ(first.name, "left_hip_pitch_joint");
  EXPECT_DOUBLE_EQ(first.q_low, -2.5307);
  EXPECT_DOUBLE_EQ(first.q_high, 2.8798);
  EXPECT_DOUBLE_EQ(first.dq_max, 32.0);
  EXPECT_DOUBLE_EQ(first.tau_max, 88.0);
  const JointLimit & last = p.joints.back();
  EXPECT_EQ(last.name, "right_wrist_yaw_joint");
  EXPECT_DOUBLE_EQ(last.q_low, -1.614429558);
  EXPECT_DOUBLE_EQ(last.q_high, 1.614429558);
  EXPECT_DOUBLE_EQ(last.dq_max, 22.0);
  EXPECT_DOUBLE_EQ(last.tau_max, 5.0);
}

TEST(RobotProfile, G1HasNoArmSdk)
{
  const RobotProfile p = golem_safety::load_robot_profile("g1");
  EXPECT_EQ(p.modes, (std::vector<Mode>{Mode::Split, Mode::FullBody}));
  // No slot past the body joints carries a value: the file has these keys and no other.
  const YAML::Node root = YAML::LoadFile(
    ament_index_cpp::get_package_share_directory("golem_safety") + "/robots/g1.yaml");
  std::set<std::string> keys;
  for (const auto & entry : root) {
    keys.insert(entry.first.as<std::string>());
  }
  EXPECT_EQ(
    keys, (std::set<std::string>{"joints", "lower_slots", "mode_machine", "mode_pr", "modes",
      "motor_slots", "robot", "upper_slots"}));
}

TEST(RobotProfile, UnknownRobotThrows)
{
  const std::string message =
    error_of([] {golem_safety::load_robot_profile("no_such_robot");});
  EXPECT_NE(message.find("no_such_robot"), std::string::npos) << message;
}

TEST(RobotProfile, ParsesAMinimalProfile)
{
  const RobotProfile p = golem_safety::parse_robot_profile(kValid, "toy.yaml");
  EXPECT_EQ(p.robot, "toy");
  EXPECT_EQ(p.mode_machine, 7);
  ASSERT_EQ(p.joints.size(), 2u);
  EXPECT_EQ(p.joints[1].name, "b_joint");
  EXPECT_DOUBLE_EQ(p.joints[1].q_low, -2.0);
  EXPECT_DOUBLE_EQ(p.joints[1].q_high, 2.0);
  EXPECT_DOUBLE_EQ(p.joints[1].dq_max, 11.0);
  EXPECT_DOUBLE_EQ(p.joints[1].tau_max, 21.0);
  EXPECT_EQ(p.modes, (std::vector<Mode>{Mode::Split}));
}

TEST(RobotProfile, ErrorsNameTheSourceAndTheKey)
{
  const std::string message = parse_error(replaced(kValid, "mode_machine: 7\n", ""));
  EXPECT_NE(message.find("toy.yaml"), std::string::npos) << message;
  EXPECT_NE(message.find("mode_machine"), std::string::npos) << message;
}

TEST(RobotProfile, UnknownKeyThrows)
{
  const std::string message =
    parse_error(replaced(kValid, "mode_pr: 0\n", "mode_pr: 0\nmodee: 1\n"));
  EXPECT_NE(message.find("modee"), std::string::npos) << message;
}

TEST(RobotProfile, SlotMustEqualPositionInTheJointList)
{
  const std::string message = parse_error(replaced(kValid, "slot: 1,", "slot: 3,"));
  EXPECT_NE(message.find("b_joint"), std::string::npos) << message;
  EXPECT_NE(message.find("slot"), std::string::npos) << message;
}

TEST(RobotProfile, InvertedPositionLimitsThrow)
{
  const std::string message =
    parse_error(replaced(kValid, "position: [-2.0, 2.0]", "position: [2.0, -2.0]"));
  EXPECT_NE(message.find("b_joint"), std::string::npos) << message;
  EXPECT_NE(message.find("position"), std::string::npos) << message;
}

TEST(RobotProfile, NonFiniteLimitThrows)
{
  const std::string message =
    parse_error(replaced(kValid, "velocity: 11.0", "velocity: .nan"));
  EXPECT_NE(message.find("velocity"), std::string::npos) << message;
}

TEST(RobotProfile, SlotRangeMustLieInsideTheBodyJoints)
{
  const std::string message =
    parse_error(replaced(kValid, "upper_slots: [1, 1]", "upper_slots: [1, 2]"));
  EXPECT_NE(message.find("upper_slots"), std::string::npos) << message;
}

TEST(RobotProfile, LowerAndUpperSlotsMustNotOverlap)
{
  const std::string message =
    parse_error(replaced(kValid, "upper_slots: [1, 1]", "upper_slots: [0, 1]"));
  EXPECT_NE(message.find("overlap"), std::string::npos) << message;
}

TEST(RobotProfile, MotorSlotsMustBeTheLowCmdArray)
{
  for (const char * slots : {"34", "36"}) {
    const std::string message =
      parse_error(replaced(kValid, "motor_slots: 35", std::string("motor_slots: ") + slots));
    EXPECT_NE(message.find("motor_slots"), std::string::npos) << message;
    EXPECT_NE(message.find("35"), std::string::npos) << message;
  }
}

TEST(RobotProfile, JointsMustFitTheLowCmdArray)
{
  // 36 body joints, one more than the LowCmd array holds.
  std::string joints = "joints:\n";
  for (int slot = 0; slot < 36; ++slot) {
    joints += "  - {name: j" + std::to_string(slot) + ", slot: " + std::to_string(slot) +
      ", position: [-1.0, 1.0], velocity: 10.0, effort: 20.0}\n";
  }
  std::string text = kValid;
  text = replaced(text, "upper_slots: [1, 1]", "upper_slots: [1, 35]");
  text = text.substr(0, text.find("joints:")) + joints;
  const std::string message = parse_error(text);
  EXPECT_NE(message.find("36 body joints"), std::string::npos) << message;
}

TEST(RobotProfile, LowerAndUpperSlotsMustCoverEveryBodyJoint)
{
  const std::string three_joints = replaced(
    kValid, "effort: 21.0}\n",
    "effort: 21.0}\n  - {name: c_joint, slot: 2, position: [-1.0, 1.0], velocity: 10.0, "
    "effort: 20.0}\n");
  const std::string message = parse_error(three_joints);
  EXPECT_NE(message.find("lower_slots"), std::string::npos) << message;
  EXPECT_NE(message.find("upper_slots"), std::string::npos) << message;
  EXPECT_NE(message.find("c_joint"), std::string::npos) << message;
  // the same profile with the upper range over both upper joints parses
  EXPECT_NO_THROW(
    golem_safety::parse_robot_profile(
      replaced(three_joints, "upper_slots: [1, 1]", "upper_slots: [1, 2]"), "toy.yaml"));
}

TEST(RobotProfile, AModeListedTwiceThrows)
{
  const std::string message = parse_error(
    replaced(kValid, "modes: [split_mode]", "modes: [split_mode, full_body_mode, split_mode]"));
  EXPECT_NE(message.find("split_mode"), std::string::npos) << message;
  EXPECT_NE(message.find("twice"), std::string::npos) << message;
}

TEST(Mode, ParsesTheTwoModeNames)
{
  EXPECT_EQ(golem_safety::parse_mode("full_body_mode"), Mode::FullBody);
  EXPECT_EQ(golem_safety::parse_mode("split_mode"), Mode::Split);
}

TEST(Mode, UnknownNameThrows)
{
  const std::string message = error_of([] {golem_safety::parse_mode("full");});
  EXPECT_NE(message.find("full"), std::string::npos) << message;
}

TEST(SlotRange, ContainsBothEnds)
{
  const SlotRange r{12, 26};
  EXPECT_FALSE(r.contains(11));
  EXPECT_TRUE(r.contains(12));
  EXPECT_TRUE(r.contains(26));
  EXPECT_FALSE(r.contains(27));
}
