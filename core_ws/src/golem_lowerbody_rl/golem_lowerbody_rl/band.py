"""
Decide when the leg node releases the sim's elastic band, with no ROS.

The policy must be running, and frame_task_server up (its first /left_ee_pose) or
band.max_wait passed since the policy started.
"""

__all__ = ['BandRelease']


class BandRelease:
    """
    Ask for the sim's /elastic_band/toggle once, and take its reply as final.

    SAFETY: the toggle flips the band, so a second call would hang a free robot from it
    again or drop a held one. poll() asks for the one call when the release is due and the
    service is there (never while it is not); result() takes how it ended. A success
    marks the band released, a failure leaves it on, and either way nothing is asked
    again. A call stays in flight until its result: there is no timeout after which a
    late reply could meet a second call. Disabled (the real robots, which have no band),
    it never asks.

    Not thread-safe: the caller serialises the calls. Times are s of robot time (the sim's
    clock under use_sim_time, the steady clock on the robots).
    """

    def __init__(self, enabled: bool, max_wait: float):
        """Ask only when enabled; max_wait s of robot time after a start with no /left_ee_pose."""
        self.enabled = enabled
        self.max_wait = max_wait
        self.released = False
        # a call is due but the service is not there yet (for the caller's log)
        self.service_missing = False
        self._in_flight = False
        self._called = False  # the one call has been asked for

    @property
    def in_flight(self) -> bool:
        """Return whether the one call has been asked for and has no result yet."""
        return self._in_flight

    def poll(self, now: float, running_since: float | None, ee_pose_seen: bool,
             service_ready: bool) -> str | None:
        """
        Return the reason to call /elastic_band/toggle now, and mark the call in flight.

        None when no call is due, and for good once it has been asked for. running_since
        is when the policy started (None while it does not run), ee_pose_seen whether a
        /left_ee_pose has come, service_ready whether the toggle service is there.
        """
        if not self.enabled or self._called or running_since is None:
            return None
        if ee_pose_seen:
            reason = 'the policy runs and frame_task_server is up (/left_ee_pose)'
        elif now - running_since >= self.max_wait:
            reason = (f'the policy has run for {self.max_wait:.0f} s, the maximum wait '
                      '(band.max_wait), with no /left_ee_pose from frame_task_server')
        else:
            return None
        self.service_missing = not service_ready
        if not service_ready:
            return None
        self._in_flight = self._called = True
        return reason

    def result(self, released: bool) -> bool:
        """
        Take how the one call ended: released true if it took the band off.

        Return whether this was the awaited result; any other (none in flight, a late or
        repeated reply) changes nothing.
        """
        if not self._in_flight:
            return False
        self._in_flight = False
        self.released = released
        return True
