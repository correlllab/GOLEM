#pragma once

#include <cerrno>
#include <cmath>
#include <ctime>

namespace golem_safety
{

// Seconds on CLOCK_BOOTTIME: monotonic, so a wall-clock step neither fires nor hides a
// timeout, and counting the time the host was suspended, so a suspend is a gap.
inline double boot_time()
{
  timespec ts{};
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// Sleeps until boot_time() reaches `deadline`; returns at once for a deadline passed.
inline void sleep_until_boot_time(double deadline)
{
  if (!(deadline > boot_time())) {
    return;
  }
  timespec ts{};
  double whole = 0.0;
  const double fraction = std::modf(deadline, &whole);
  ts.tv_sec = static_cast<time_t>(whole);
  ts.tv_nsec = static_cast<long>(fraction * 1e9);
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec += 1;
    ts.tv_nsec -= 1000000000L;
  }
  while (clock_nanosleep(CLOCK_BOOTTIME, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
  }
}

}  // namespace golem_safety
