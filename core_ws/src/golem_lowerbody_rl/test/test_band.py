"""
Check when the sim's elastic band is released, with no ROS.

/elastic_band/toggle flips the band, so BandRelease asks for it once and takes the reply
as final.
"""

from golem_lowerbody_rl.band import BandRelease

MAX_WAIT = 30.0
START = 100.0  # when the policy started (running_since), s


def test_band_release_waits_for_the_policy():
    """Ask for nothing before the policy runs, then once /left_ee_pose has come."""
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    for now in (START, START + 2 * MAX_WAIT):
        assert band.poll(now, None, ee_pose_seen=True, service_ready=True) is None
    reason = band.poll(START + 0.1, START, ee_pose_seen=True, service_ready=True)
    assert reason is not None and '/left_ee_pose' in reason
    assert not band.released
    # the call is in flight: its reply is the awaited result
    assert band.result(True) and band.released


def test_band_release_waits_for_frame_task_or_max_wait():
    """Ask max_wait after the policy started when no /left_ee_pose has come."""
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    for now in (START, START + 1.0, START + MAX_WAIT - 0.01):
        assert band.poll(now, START, ee_pose_seen=False, service_ready=True) is None
    reason = band.poll(START + MAX_WAIT + 0.01, START, ee_pose_seen=False, service_ready=True)
    assert reason is not None and 'maximum wait' in reason and '30 s' in reason


def test_band_release_waits_for_the_service():
    """Ask only once the service is there, and say that it is missing meanwhile."""
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, ee_pose_seen=True, service_ready=False) is None
    # nothing is in flight: a reply is not taken
    assert band.service_missing and not band.result(True)
    assert band.poll(START + 1.1, START, ee_pose_seen=True, service_ready=True) is not None


def test_band_release_one_call_in_flight():
    """Never ask again while the one call waits for its reply, however long."""
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, ee_pose_seen=True, service_ready=True) is not None
    for now in (START + 1.1, START + 5.0, START + 10 * MAX_WAIT):
        assert band.poll(now, START, ee_pose_seen=True, service_ready=True) is None
    assert not band.released
    assert band.result(True)
    # nothing is in flight any more: a repeated reply changes nothing
    assert band.released and not band.result(False) and band.released


def test_failed_release_is_not_retried():
    """Take a failed reply as final: nothing is asked again and a late success is ignored."""
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, ee_pose_seen=True, service_ready=True) is not None
    assert band.result(False)
    assert not band.released
    for now in (START + 1.1, START + 2 * MAX_WAIT):
        assert band.poll(now, START, ee_pose_seen=True, service_ready=True) is None
    assert not band.result(True)
    assert not band.released


def test_band_release_never_after_success():
    """Ignore late or repeated replies after a success, and never ask again."""
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, ee_pose_seen=True, service_ready=True) is not None
    assert band.result(True)
    assert not band.result(False)
    assert not band.result(True)
    assert band.released
    for now in (START + 1.1, START + 2 * MAX_WAIT):
        for seen in (True, False):
            assert band.poll(now, START, ee_pose_seen=seen, service_ready=True) is None
    assert band.released


def test_band_release_disabled_never_calls():
    """Never ask with the band disabled (the real robots have none)."""
    band = BandRelease(enabled=False, max_wait=MAX_WAIT)
    for now in (START, START + 1.0, START + 2 * MAX_WAIT):
        for running_since in (None, START):
            assert band.poll(now, running_since, ee_pose_seen=True, service_ready=True) is None
    assert not band.released and not band.service_missing
    assert not band.result(True)
