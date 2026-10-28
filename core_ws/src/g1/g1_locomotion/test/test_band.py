'''
The sim's band release with no ROS: when /elastic_band/toggle is called, and that the
toggle (which flips the band) is never sent twice
'''

from g1_locomotion.band import BandRelease

MAX_WAIT = 30.0
START = 100.0  # when the policy started (walking_since), s


def test_band_release_waits_for_walking():
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    # before the policy runs, nothing, even with frame_task_server up and long past max_wait
    for now in (START, START + 2 * MAX_WAIT):
        assert band.poll(now, None, frame_task_ready=True, service_ready=True) is None
    reason = band.poll(START + 0.1, START, frame_task_ready=True, service_ready=True)
    assert reason is not None and '/left_ee_pose' in reason
    assert not band.released


def test_band_release_waits_for_frame_task():
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    for now in (START, START + 1.0, START + MAX_WAIT - 0.01):
        assert band.poll(now, START, frame_task_ready=False, service_ready=True) is None
    reason = band.poll(START + MAX_WAIT + 0.01, START, frame_task_ready=False, service_ready=True)
    assert reason is not None and 'maximum wait' in reason and '30 s' in reason


def test_band_release_waits_for_the_service():
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, frame_task_ready=True, service_ready=False) is None
    assert band.service_missing
    assert band.poll(START + 1.1, START, frame_task_ready=True, service_ready=True) is not None


def test_band_release_one_call_in_flight():
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, frame_task_ready=True, service_ready=True) is not None
    # no response yet: never a second call, however long it takes
    for now in (START + 1.1, START + 5.0, START + 10 * MAX_WAIT):
        assert band.poll(now, START, frame_task_ready=True, service_ready=True) is None
    assert not band.released
    band.result(True)
    assert band.released


def test_band_release_retries_after_failure():
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, frame_task_ready=True, service_ready=True) is not None
    band.result(False)
    assert not band.released
    assert band.poll(START + 1.1, START, frame_task_ready=True, service_ready=True) is not None
    band.result(True)
    assert band.released


def test_band_release_never_after_success():
    band = BandRelease(enabled=True, max_wait=MAX_WAIT)
    assert band.poll(START + 1.0, START, frame_task_ready=True, service_ready=True) is not None
    band.result(True)
    # a late or repeated response changes nothing, and nothing is ever sent again
    band.result(False)
    band.result(True)
    assert band.released
    for now in (START + 1.1, START + 2 * MAX_WAIT):
        for ready in (True, False):
            assert band.poll(now, START, frame_task_ready=ready, service_ready=True) is None
    assert band.released


def test_band_release_disabled_never_calls():
    band = BandRelease(enabled=False, max_wait=MAX_WAIT)
    for now in (START, START + 1.0, START + 2 * MAX_WAIT):
        for walking_since in (None, START):
            assert band.poll(now, walking_since, frame_task_ready=True, service_ready=True) is None
    assert not band.released and not band.service_missing
