#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "golem_safety/boot_clock.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/time.hpp"

namespace golem_safety
{

// Tell and wait robot time, the clock the relay's rate and every SafetyCore timing count: the
// boot clock (boot_clock.hpp) on the real robots, and the node's ROS clock, which follows a
// simulator's /clock, under use_sim_time. On the ROS clock, robot time advances by each
// forward step of /clock, stands still while /clock does, and never goes back: a /clock that
// steps back (a restarted simulator) leaves it where it was, so SafetyCore's `now` never
// goes backwards.
// Threads: now(), started() and sleep_until() from one thread; the jump callback that wakes
// a sleep runs on the thread that takes /clock.
class RobotClock
{
public:
  // The longest a robot-time sleep goes without checking its stop, of steady time.
  static constexpr std::chrono::milliseconds kWaitSlice{10};

  // Use the boot clock.
  RobotClock() = default;

  // Use `ros_clock`, an RCL_ROS_TIME clock that /clock drives; a sleep wakes on its every
  // change.
  explicit RobotClock(rclcpp::Clock::SharedPtr ros_clock)
  : ros_clock_(std::move(ros_clock))
  {
    rcl_jump_threshold_t threshold{};
    threshold.on_clock_change = true;
    threshold.min_forward.nanoseconds = 1;
    threshold.min_backward.nanoseconds = -1;
    jump_handler_ = ros_clock_->create_jump_callback(
      [] {}, [this](const rcl_time_jump_t &) {wake();}, threshold);
  }

  RobotClock(const RobotClock &) = delete;
  RobotClock & operator=(const RobotClock &) = delete;

  // Return whether robot time is a simulator's /clock.
  bool sim_time() const {return ros_clock_ != nullptr;}

  // Return whether robot time has started: a first /clock has arrived (always, on the boot
  // clock).
  bool started() const {return !ros_clock_ || ros_clock_->now().nanoseconds() > 0;}

  // Return robot time, s.
  double now()
  {
    if (!ros_clock_) {
      return boot_time();
    }
    const int64_t ros_ns = ros_clock_->now().nanoseconds();
    if (!last_ros_ns_) {
      time_ns_ = ros_ns;
    } else if (ros_ns > *last_ros_ns_) {
      time_ns_ += ros_ns - *last_ros_ns_;
    }
    last_ros_ns_ = ros_ns;
    return static_cast<double>(time_ns_) * 1e-9;
  }

  // Wait until robot time reaches `deadline`, s. On the boot clock this is
  // sleep_until_boot_time(deadline), and `stop` is not read. On /clock it also ends once
  // stop() returns true, which is checked at least every kWaitSlice of steady time, so a
  // stopped /clock cannot hold the caller for ever.
  template<class Stop>
  void sleep_until(double deadline, Stop && stop)
  {
    if (!ros_clock_) {
      sleep_until_boot_time(deadline);
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop() && now() < deadline) {
      wake_.wait_for(lock, kWaitSlice);
    }
  }

  // Wait up to kWaitSlice of steady time, or until /clock moves.
  void wait_slice()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    wake_.wait_for(lock, kWaitSlice);
  }

private:
  // Wake a sleep: /clock has moved.
  void wake()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
    }
    wake_.notify_all();
  }

  const rclcpp::Clock::SharedPtr ros_clock_;
  std::optional<int64_t> last_ros_ns_;  // the ROS time now() last read, ns
  int64_t time_ns_ = 0;  // robot time, ns
  std::mutex mutex_;
  std::condition_variable wake_;
  // Last, so it is destroyed first: no callback runs on the members above once they are gone.
  rclcpp::JumpHandler::SharedPtr jump_handler_;
};

}  // namespace golem_safety
