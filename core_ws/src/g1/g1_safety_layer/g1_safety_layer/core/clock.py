'''The clock every watchdog and weight ramp is timed with'''

import time

# CLOCK_BOOTTIME (Linux) keeps counting while the laptop is suspended and is
# never stepped. time.time() can be stepped, by seconds or hours (date, or
# systemd-timesyncd for offsets above 0.4 s), which would fire a watchdog or
# hide a real gap. time.monotonic() (CLOCK_MONOTONIC) is not stepped either,
# but it stops during a suspend, so a gap that spans one would look short on
# resume. Both are still slewed with the wall clock: by at most 500 ppm through
# the kernel's NTP frequency (50 us over a 0.1 s watchdog), more under chrony
_BOOTTIME = getattr(time, 'CLOCK_BOOTTIME', None)


def watchdog_time() -> float:
    '''Seconds on CLOCK_BOOTTIME, or time.monotonic() where it does not exist'''
    if _BOOTTIME is None:
        return time.monotonic()
    return time.clock_gettime(_BOOTTIME)
