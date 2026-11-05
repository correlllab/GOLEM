#include <gtest/gtest.h>

#include <time.h>

#include "golem_safety/boot_clock.hpp"

namespace
{

// Return `clock` in s.
double read(clockid_t clock)
{
  timespec ts{};
  clock_gettime(clock, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

}  // namespace

// boot_time() reads CLOCK_BOOTTIME.
TEST(BootClock, IsClockBoottime)
{
  const double before = read(CLOCK_BOOTTIME);
  const double now = golem_safety::boot_time();
  const double after = read(CLOCK_BOOTTIME);
  EXPECT_LE(before, now);
  EXPECT_LE(now, after);
}

// sleep_until_boot_time() returns at or after the deadline, and at once for a past one.
TEST(BootClock, SleepUntilWaitsForTheDeadline)
{
  const double deadline = golem_safety::boot_time() + 0.02;
  golem_safety::sleep_until_boot_time(deadline);
  EXPECT_GE(golem_safety::boot_time(), deadline);
  const double start = golem_safety::boot_time();
  golem_safety::sleep_until_boot_time(start - 1.0);
  EXPECT_LT(golem_safety::boot_time() - start, 0.01);
}
