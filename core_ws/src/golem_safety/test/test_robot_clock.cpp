#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include "golem_safety/boot_clock.hpp"
#include "golem_safety/robot_clock.hpp"
#include "rcl/time.h"
#include "rclcpp/clock.hpp"

namespace
{

using golem_safety::RobotClock;
using golem_safety::boot_time;

// Drive an RCL_ROS_TIME clock by hand, as a TimeSource does on each /clock.
class FakeSimClock
{
public:
  // Make the clock and turn its override on, at ROS time 0 (no /clock yet).
  FakeSimClock()
  : clock(std::make_shared<rclcpp::Clock>(RCL_ROS_TIME))
  {
    EXPECT_EQ(rcl_enable_ros_time_override(clock->get_clock_handle()), RCL_RET_OK);
  }

  // Set ROS time to `seconds`, as one /clock message does.
  void set(double seconds)
  {
    EXPECT_EQ(
      rcl_set_ros_time_override(
        clock->get_clock_handle(), static_cast<int64_t>(seconds * 1e9)), RCL_RET_OK);
  }

  rclcpp::Clock::SharedPtr clock;
};

}  // namespace

// Without use_sim_time robot time is the boot clock, and a sleep is sleep_until_boot_time.
TEST(RobotClock, BootClockByDefault)
{
  RobotClock clock;
  EXPECT_FALSE(clock.sim_time());
  EXPECT_TRUE(clock.started());
  const double before = boot_time();
  const double now = clock.now();
  EXPECT_LE(before, now);
  EXPECT_LE(now, boot_time());
  const double deadline = boot_time() + 0.02;
  clock.sleep_until(deadline, [] {return true;});  // the stop is not read on the boot clock
  EXPECT_GE(boot_time(), deadline);
}

// On /clock, robot time is ROS time, and it starts with the first /clock.
TEST(RobotClock, FollowsTheRosClock)
{
  FakeSimClock sim;
  RobotClock clock(sim.clock);
  EXPECT_TRUE(clock.sim_time());
  EXPECT_FALSE(clock.started());
  sim.set(100.0);
  EXPECT_TRUE(clock.started());
  EXPECT_DOUBLE_EQ(clock.now(), 100.0);
  sim.set(100.25);
  EXPECT_DOUBLE_EQ(clock.now(), 100.25);
}

// A /clock that steps back leaves robot time where it was; it advances again from there.
TEST(RobotClock, NeverGoesBack)
{
  FakeSimClock sim;
  RobotClock clock(sim.clock);
  sim.set(100.0);
  EXPECT_DOUBLE_EQ(clock.now(), 100.0);
  sim.set(3.0);
  EXPECT_DOUBLE_EQ(clock.now(), 100.0);
  sim.set(3.5);
  EXPECT_DOUBLE_EQ(clock.now(), 100.5);
}

// A sleep on /clock ends when /clock reaches the deadline, however much steady time passes.
TEST(RobotClock, SleepEndsOnTheSimDeadline)
{
  FakeSimClock sim;
  RobotClock clock(sim.clock);
  sim.set(100.0);
  std::atomic<bool> done{false};
  std::thread sleeper([&] {
      clock.sleep_until(100.5, [] {return false;});
      done = true;
    });
  // /clock at a quarter of the wall clock's speed: 0.2 s of steady time is 0.05 s of sim
  for (int step = 1; step <= 8; ++step) {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    sim.set(100.0 + 0.0125 * step);
  }
  EXPECT_FALSE(done) << "the sleep ended before the sim deadline";
  // /clock stopped: the sleep stays
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(done);
  const double woken_at = boot_time();
  sim.set(100.5);
  sleeper.join();
  EXPECT_TRUE(done);
  // woken by /clock itself, well within a wait slice
  EXPECT_LT(boot_time() - woken_at, 0.009);
}

// A sleep on a stopped /clock ends once its stop returns true.
TEST(RobotClock, StopEndsASleepOnAStoppedClock)
{
  FakeSimClock sim;
  RobotClock clock(sim.clock);
  sim.set(100.0);
  std::atomic<bool> stop{false};
  std::atomic<bool> done{false};
  std::thread sleeper([&] {
      clock.sleep_until(200.0, [&] {return stop.load();});
      done = true;
    });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(done);
  const double stopped_at = boot_time();
  stop = true;
  sleeper.join();
  EXPECT_LT(boot_time() - stopped_at, 0.05);
}
