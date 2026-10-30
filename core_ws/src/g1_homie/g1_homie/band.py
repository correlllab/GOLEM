'''
When the walking node releases the sim's elastic band, with no ROS: the policy must be
running, and frame_task_server up or band.max_wait passed since the policy started
'''


class BandRelease:
    '''
    The sim's /elastic_band/toggle flips the band, so it is called until one call
    succeeds and never again after that: at most one call in flight, a call stays in
    flight until its result (there is no timeout, after which a late response could
    meet a second call), and a failed one is sent again. Disabled (the real robot,
    which has no band), it never calls.

    The caller calls poll() often (the node at 10 Hz); when it returns a reason, the
    caller sends the call at once and reports how it ended with result(). Not
    thread-safe: the caller serializes the calls. Times are one monotonic clock, in s
    '''

    def __init__(self, enabled: bool, max_wait: float):
        self.enabled = enabled
        self.max_wait = max_wait
        self.released = False
        # a call is due but the service is not there yet (for the caller's log)
        self.service_missing = False
        self._in_flight = False

    def poll(self, now: float, walking_since: float | None, frame_task_ready: bool,
             service_ready: bool) -> str | None:
        '''
        The reason to call /elastic_band/toggle now, and the call marked in flight; None
        when no call is due. walking_since is when the policy started (None while it
        does not run), frame_task_ready whether /left_ee_pose has been seen
        '''
        if not self.enabled or self.released or self._in_flight or walking_since is None:
            return None
        if frame_task_ready:
            reason = 'the policy runs and frame_task_server is up (/left_ee_pose)'
        elif now - walking_since >= self.max_wait:
            reason = (f'the policy has run for {self.max_wait:.0f} s, the maximum wait '
                      '(band.max_wait), with no /left_ee_pose from frame_task_server')
        else:
            return None
        self.service_missing = not service_ready
        if not service_ready:
            return None
        self._in_flight = True
        return reason

    def result(self, success: bool) -> None:
        '''How the call poll() asked for ended: success, or failed and to be sent again'''
        if not self._in_flight:
            return
        self._in_flight = False
        self.released = success
