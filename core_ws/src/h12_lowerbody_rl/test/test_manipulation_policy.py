"""Offline numerical contracts only. Never import a ROS node or initialize DDS."""

import ast
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import MethodType, SimpleNamespace
import time
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from h12_lowerbody_rl.manipulation_policy import AlmiManipStandPolicy, FameFinetunePolicy
from h12_lowerbody_rl.manipulation_policy import _ManipStandActor
from h12_lowerbody_rl.policy import (
    AlmiPolicy, FamePolicy, WalkPolicy, RobotState, apply_imu_offset, imu_offset_quat,
    quat_rotate_inverse,
)
from h12_lowerbody_rl.rma import EnvFactorEncoder
from h12_lowerbody_rl.policy_manager import PolicyManager


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "policies"
VARIANTS = [(FameFinetunePolicy, "fame_finetune"),
            (AlmiManipStandPolicy, "almi_manip_stand")]
torch.set_num_threads(1)


def load(cls, name):
    return cls(str(POLICIES / name / (name + ".yaml")))


def sample(step):
    q = np.arange(27, dtype=np.float32) * .017 - .21 + step * .003
    dq = np.arange(27, dtype=np.float32) * -.023 + .4 + step * .007
    quat = np.array([.91, .12, -.18, .24], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    return RobotState(q, dq, quat, np.array([.21, -.13, .37], dtype=np.float32),
                      np.array([.7, -.2, .3], dtype=np.float32), .3, step * .02)


def reference_imu(state):
    w, x, y, z = state.quat / np.linalg.norm(state.quat)
    torso_rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    c, s = np.cos(state.q[12]), np.sin(state.q[12])
    waist = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pelvis_rotation = torso_rotation @ waist.T
    gyro = waist @ state.gyro - [0, 0, state.dq[12]]
    return gyro.astype(np.float32), (pelvis_rotation.T @ [0, 0, -1]).astype(np.float32)


def test_fame_actual_checkpoint_replay_matches_upstream_contract():
    policy = load(*VARIANTS[0])
    root = POLICIES / "fame_finetune"
    actor = torch.jit.load(str(root / "policy_4599.pt"), map_location="cpu").eval()
    encoder = EnvFactorEncoder()
    encoder.load_state_dict(torch.load(root / "encoder_4599.pt",
                                      map_location="cpu", weights_only=True))
    encoder.eval()
    frames = deque(maxlen=3)
    latents = np.zeros((3, 8), dtype=np.float32)
    previous = np.zeros(12, dtype=np.float32)
    policy.reset(sample(0))
    defaults = np.r_[policy.nominal_lower, np.zeros(15)].astype(np.float32)
    for step in range(12):
        state = sample(step)
        omega, gravity = reference_imu(state)
        frame = np.concatenate((np.zeros(3), [1.], omega * .25, gravity,
                                state.q - defaults, state.dq * .05, previous)).astype(np.float32)
        frames.append(frame)
        proprio = np.zeros((3, 76), dtype=np.float32)
        proprio[:len(frames)] = np.asarray(frames)
        factors = np.r_[state.q[12:], np.zeros(6)].astype(np.float32)
        with torch.no_grad():
            latent = encoder(torch.from_numpy(factors)[None]).numpy()[0]
            latents[:-1] = latents[1:].copy()
            latents[-1] = latent
            observation = np.r_[proprio.ravel(), latents.ravel()]
            expected_action = actor(torch.from_numpy(observation)[None]).numpy()[0]
        actual = policy.compute(state)
        np.testing.assert_allclose(policy._action, expected_action, atol=3e-6, rtol=2e-6)
        expected_q = np.clip(policy.nominal_lower + expected_action * .25,
                             policy._lower_limit, policy._upper_limit)
        np.testing.assert_allclose(actual.target_q, expected_q, atol=1e-6)
        previous = expected_action


def test_almi_actual_checkpoint_recurrent_replay_matches_upstream_contract():
    policy = load(*VARIANTS[1])
    path = POLICIES / "almi_manip_stand/model_almi_manip_stand.pt"
    state_dict = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
    rnn = nn.LSTM(65, 64, 1).eval()
    rnn.load_state_dict({k.removeprefix("rnn.rnn."): v for k, v in state_dict.items()
                         if k.startswith("rnn.rnn.")})
    actor = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 32),
                          nn.ELU(), nn.Linear(32, 12)).eval()
    actor.load_state_dict({k.removeprefix("mlp."): v for k, v in state_dict.items()
                           if k.startswith("mlp.")})
    previous = np.zeros(12, dtype=np.float32)
    hidden = None
    ids = list(range(17)) + list(range(20, 24))
    policy.reset(sample(0))
    for step in range(20):
        state = sample(step)
        omega, gravity = reference_imu(state)
        obs = np.concatenate((omega * .25, gravity, np.zeros(3),
                              state.q[ids] - policy._default_angles,
                              state.dq[ids] * .05, previous, np.zeros(2))).astype(np.float32)
        with torch.no_grad():
            latent, hidden = rnn(torch.from_numpy(obs)[None, None], hidden)
            expected_action = actor(latent[0]).numpy()[0]
        actual = policy.compute(state)
        np.testing.assert_allclose(policy._obs, obs, atol=3e-7)
        np.testing.assert_allclose(policy._action, expected_action, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(actual.target_q,
                                   np.clip(policy.nominal_lower + .25 * expected_action,
                                           policy._lower_limit, policy._upper_limit), atol=1e-6)
        previous = expected_action


@pytest.mark.parametrize("cls,name", VARIANTS)
def test_reset_is_reproducible_and_commands_cannot_enable_walking(cls, name):
    policy = load(cls, name)
    state = sample(0)
    policy.reset(state)
    first = policy.compute(state).target_q.copy()
    for step in range(1, 8):
        policy.compute(sample(step))
    policy.reset(state)
    second = policy.compute(replace(state, cmd=np.zeros(3, dtype=np.float32),
                                    height_cmd=1.)).target_q
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("cls,name", VARIANTS)
@pytest.mark.parametrize("field,value", [
    ("q", np.zeros(26)), ("dq", np.full(27, np.nan)),
    ("quat", np.zeros(4)), ("gyro", np.full(3, np.inf)),
    ("cmd", np.full(3, np.nan)), ("t", np.nan), ("height_cmd", np.inf),
])
def test_invalid_state_rejected_before_inference(cls, name, field, value):
    policy = load(cls, name)
    with pytest.raises(ValueError):
        policy.compute(replace(sample(0), **{field: value}))


@pytest.mark.parametrize("cls,name", VARIANTS)
def test_artifact_provenance_and_corruption_fail_closed(cls, name, tmp_path, monkeypatch):
    root = POLICIES / name
    cfg = yaml.safe_load((root / (name + ".yaml")).read_text())
    provenance = yaml.safe_load((root / "SOURCE.yaml").read_text())
    assert provenance["source_revision"] == "c946b1a74f6d6812d82f80fdc7651acb2840acd2"
    for key in ("policy_path", "encoder_path"):
        if key in cfg:
            path = root / cfg[key]
            digest = sha256(path.read_bytes()).hexdigest()
            assert digest == provenance["artifacts"][path.name]["sha256"]
            cfg[key] = str(path)
    if name == "fame_finetune":
        assert provenance["source_original_license_status"] == (
            "not established from upstream checkpoint metadata")
    cfg["artifact_sha256"]["policy_path"] = "0" * 64
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded corrupt file"))
    monkeypatch.setattr(torch.jit, "load", lambda *a, **kw: pytest.fail("loaded corrupt file"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cls(str(config_path))


@pytest.mark.parametrize("cls,name", VARIANTS)
def test_raw_action_retained_but_targets_clipped_and_nan_rejected(cls, name):
    policy = load(cls, name)

    class ExtremeActor(nn.Module):
        def forward(self, obs):
            return torch.full((1, 12), 1000.)

    policy._policy = ExtremeActor()
    result = policy.compute(sample(0))
    np.testing.assert_array_equal(policy._action, np.full(12, 1000.))
    np.testing.assert_array_equal(result.target_q, policy._upper_limit)

    class BadActor(nn.Module):
        def forward(self, obs):
            return torch.full((1, 12), float("nan"))

    policy._policy = BadActor()
    with pytest.raises(ValueError, match="invalid policy output"):
        policy.compute(sample(1))


def test_legacy_names_and_fame_history_unchanged():
    assert FamePolicy.name == "fame"
    assert AlmiPolicy.name == "almi"
    legacy = FamePolicy(str(POLICIES / "fame/fame.yaml"))
    legacy.reset(sample(0))
    assert len(legacy._obs_history) == 3
    assert all(np.count_nonzero(frame) == 0 for frame in legacy._obs_history)


@pytest.mark.parametrize("cls,name", VARIANTS)
@pytest.mark.parametrize("key,value", [("kps", [1.] * 11), ("kds", [-1.] * 12),
                                      ("action_scale", .5), ("num_obs", 999)])
def test_invalid_runtime_metadata_rejected(cls, name, key, value, tmp_path):
    root = POLICIES / name
    cfg = yaml.safe_load((root / (name + ".yaml")).read_text())
    for path_key in ("policy_path", "encoder_path"):
        if path_key in cfg:
            cfg[path_key] = str(root / cfg[path_key])
    cfg[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        cls(str(path))


@pytest.mark.parametrize("mismatch", ["normalizer", "shape"])
def test_almi_actor_rejects_incompatible_checkpoint(mismatch, tmp_path):
    source = POLICIES / "almi_manip_stand/model_almi_manip_stand.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    actor = checkpoint["actor_state_dict"]
    if mismatch == "normalizer":
        actor["obs_normalizer._mean"] = torch.zeros(1, 65)
    else:
        actor["mlp.0.weight"] = torch.zeros(64, 65)
    path = tmp_path / "incompatible.pt"
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        _ManipStandActor(path)


def node_method(name):
    source = ROOT / "h12_lowerbody_rl/scripts/lowerbody_controller_node.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "LowerBodyControllerNode")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"np": np, "time": time, "LowState": object, "RobotState": RobotState,
                 "NUM_POLICY_JOINTS": 27, "apply_imu_offset": apply_imu_offset,
                 "NUM_LEG_JOINTS": 12, "PREPOSE_TOL": .15,
                 "PREPOSE_SETTLE_TICKS": 10, "PREPOSE_MAX_TICKS": 400,
                 "quat_rotate_inverse": quat_rotate_inverse,
                 "Trigger": SimpleNamespace(Request=object)}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("invalid", ["stale", "nonfinite"])
def test_node_tick_guard_without_import_or_constructor(invalid):
    errors = []
    state = sample(0)
    if invalid == "nonfinite":
        state.q[0] = np.nan
    fake = SimpleNamespace(
        _hb_dead=False, _hb_last=time.monotonic(), _hb_timeout=.5,
        _lowstate=object(), _state_from_lowstate=lambda msg: state,
        _standing_policy=load(*VARIANTS[1]), _standing_state_timeout=.1,
        _lowstate_received_at=time.monotonic() - (1. if invalid == "stale" else 0.),
        _trip_heartbeat=errors.append,
    )
    # No manager, publisher, or ROS runtime exists: guard must return before them.
    node_method("_tick")(fake)
    assert len(errors) == 1


def test_node_inference_failure_is_silent_without_import_or_constructor():
    errors = []

    def fail(state):
        raise ValueError("nonfinite policy output")

    fake = SimpleNamespace(
        _hb_dead=False, _hb_last=time.monotonic(), _hb_timeout=.5,
        _lowstate=object(), _state_from_lowstate=lambda msg: sample(0),
        _standing_policy=load(*VARIANTS[1]), _standing_state_timeout=.1,
        _lowstate_received_at=time.monotonic(), _trip_heartbeat=errors.append,
        _manager=SimpleNamespace(is_pending=lambda: False, is_idle=lambda: False, run=fail),
        _awaiting_band_release=False,
    )
    node_method("_tick")(fake)
    assert errors == ["standing policy inference failed: nonfinite policy output"]


def test_node_imu_trim_rotates_new_variant_gyro_only():
    corr = imu_offset_quat(.12, -.18, .23)
    msg = SimpleNamespace(
        motor_state=[SimpleNamespace(q=0., dq=0.) for _ in range(27)],
        imu_state=SimpleNamespace(quaternion=[1., 0., 0., 0.], gyroscope=[.3, -.2, .4]),
    )
    fake = SimpleNamespace(
        _imu_corr=corr, _standing_policy=object(), _cmd=np.zeros(3), _height_cmd=1.,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
    )
    state = node_method("_state_from_lowstate")(fake, msg)
    np.testing.assert_allclose(state.gyro, quat_rotate_inverse(corr, np.array(msg.imu_state.gyroscope)))
    fake._standing_policy = None
    legacy = node_method("_state_from_lowstate")(fake, msg)
    np.testing.assert_allclose(legacy.gyro, msg.imu_state.gyroscope)


def band_release_fixture(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    client = SimpleNamespace(ready=False, futures=[], dispatch_error=False)
    client.service_is_ready = Mock(side_effect=lambda: client.ready)

    def dispatch(request):
        if client.dispatch_error:
            raise RuntimeError("dispatch unavailable")
        future = Future()
        client.futures.append(future)
        return future

    client.call_async = Mock(side_effect=dispatch)
    logger = SimpleNamespace(info=Mock(), warn=Mock())
    manager = SimpleNamespace(is_pending=lambda: False, is_idle=lambda: False,
                              run=Mock(return_value=object()), reset_active=Mock())
    fake = SimpleNamespace(
        _hb_dead=False, _hb_last=100., _hb_timeout=100., _lowstate=object(),
        _state_from_lowstate=lambda msg: sample(0), _standing_policy=None,
        _awaiting_band_release=True, _band_released=False,
        _band_release_inflight=False, _band_retry_at=0., _band_wait_warned=False,
        _band_cli=client, _frame_task_ready=True, _request_time=None,
        _manager=manager, _publish_leg=Mock(),
        _trip_heartbeat=Mock(), get_logger=lambda: logger,
    )
    for name in ("_release_band", "_on_band_toggled", "_tick"):
        setattr(fake, name, MethodType(node_method(name), fake))
    return fake, client, clock


def test_band_release_retries_service_discovery_and_resets_only_after_disabled(monkeypatch):
    node, client, clock = band_release_fixture(monkeypatch)
    node._tick()
    clock[0] += .02
    node._tick()
    assert node._awaiting_band_release and not node._band_released
    client.call_async.assert_not_called()
    assert client.service_is_ready.call_count == 1
    node._manager.reset_active.assert_not_called()
    assert node._publish_leg.call_count == 2

    client.ready = True
    clock[0] += .6
    node._tick()
    clock[0] += .6
    node._tick()
    assert client.call_async.call_count == 1
    assert node._band_release_inflight and node._awaiting_band_release
    node._manager.reset_active.assert_not_called()

    client.futures[0].set_result(SimpleNamespace(success=True, message="disabled"))
    assert node._band_released and node._awaiting_band_release
    node._manager.reset_active.assert_not_called()
    clock[0] += .02
    node._tick()
    assert not node._awaiting_band_release
    node._manager.reset_active.assert_called_once()
    clock[0] += .02
    node._tick()
    assert client.call_async.call_count == 1
    node._manager.reset_active.assert_called_once()
    assert node._publish_leg.call_count == 6


@pytest.mark.parametrize("failure", ["dispatch", "exception", "none", "failed", "enabled", "unknown"])
def test_band_release_failure_retry_is_bounded_and_does_not_reset_policy(monkeypatch, failure):
    node, client, clock = band_release_fixture(monkeypatch)
    client.ready = True
    client.dispatch_error = failure == "dispatch"
    node._tick()
    if failure == "exception":
        client.futures[0].set_exception(RuntimeError("service failed"))
    elif failure == "none":
        client.futures[0].set_result(None)
    elif failure != "dispatch":
        message = "enabled" if failure == "enabled" else "unknown"
        if failure == "failed":
            message = "disabled"
        client.futures[0].set_result(SimpleNamespace(success=failure != "failed", message=message))
    assert not node._band_released and node._awaiting_band_release
    assert not node._band_release_inflight
    node._manager.reset_active.assert_not_called()
    clock[0] += .02
    node._tick()
    assert client.call_async.call_count == 1

    client.dispatch_error = False
    clock[0] += .6
    node._tick()
    assert client.call_async.call_count == 2
    assert node._band_release_inflight
    node._manager.reset_active.assert_not_called()
    client.futures[-1].set_result(SimpleNamespace(success=True, message="disabled"))
    clock[0] += .02
    node._tick()
    assert node._band_released and not node._awaiting_band_release
    node._manager.reset_active.assert_called_once()
    clock[0] += .02
    node._tick()
    node._manager.reset_active.assert_called_once()
    assert client.call_async.call_count == 2


def fame_startup_fixture():
    policy = load(*VARIANTS[0])
    state = RobotState(
        np.r_[policy.nominal_lower, np.zeros(15)].astype(np.float32),
        np.zeros(27, dtype=np.float32), np.array([1., 0., 0., 0.], dtype=np.float32),
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1., 0.)
    manager = PolicyManager({policy.name: policy}, log=lambda msg: None)
    manager.request(policy.name)
    for name in ("commit", "run", "reset_active"):
        setattr(manager, name, Mock(wraps=getattr(manager, name)))
    events, outputs = [], []

    def publish(command):
        events.append("publish")
        outputs.append(replace(command, target_q=command.target_q.copy()))

    fake = SimpleNamespace(
        _hb_dead=False, _hb_last=time.monotonic(), _hb_timeout=100.,
        _lowstate=object(), _state_from_lowstate=lambda msg: state,
        _standing_policy=policy, _standing_state_timeout=100.,
        _lowstate_received_at=time.monotonic(), _manager=manager,
        _prepose_start_q=None, _prepose_ticks=0, _prepose_pass=0,
        _prepose_ramp_ticks=1, _confirm_wait=True, _engage_confirmed=False,
        _awaiting_band_release=False, _band_released=False, _frame_task_ready=True,
        _request_time=None, _cmd=state.cmd, _height_cmd=1.,
        _publish_leg=publish, _publish_active=Mock(), _trip_heartbeat=Mock(),
        _release_band=Mock(side_effect=lambda reason: events.append("release")),
        get_parameter=lambda name: SimpleNamespace(value=1.),
        get_logger=lambda: SimpleNamespace(info=Mock()),
    )
    for name in ("_tick", "_clear_commands", "_on_confirm_engage"):
        setattr(fake, name, MethodType(node_method(name), fake))
    return fake, policy, state, outputs, events


def test_fame_full_startup_sequence_actual_checkpoint_no_endpoints():
    node, policy, state, outputs, events = fame_startup_fixture()
    startup = np.array([-.3700, -.2036, .1770, .3276, -.1077, -.1792,
                        -.3676, -.1076, -.1677, .3708, -.2489, .1589], dtype=np.float32)
    defaults = np.array([0., -.16, 0., .36, -.2, 0.] * 2, dtype=np.float32)
    np.testing.assert_array_equal(policy.startup_lower, startup)
    np.testing.assert_array_equal(policy.nominal_lower, defaults)
    np.testing.assert_array_equal(policy.nominal_command().target_q, defaults)
    initial = state.q[:12].copy()
    for tick in range(100):
        node._tick()
        alpha = (tick + 1) / 100
        np.testing.assert_allclose(outputs[-1].target_q,
                                   initial * (1 - alpha) + startup * alpha, atol=3e-8)
        np.testing.assert_array_equal(outputs[-1].kp, [300, 300, 300, 300, 80, 100] * 2)
        np.testing.assert_array_equal(outputs[-1].kd, [8, 8, 8, 8, 3, 5] * 2)
    # A sent target is not measured settling; hold even past timeout without confirm.
    for _ in range(410):
        node._tick()
    node._manager.commit.assert_not_called()
    node._manager.run.assert_not_called()
    node._release_band.assert_not_called()
    np.testing.assert_array_equal(outputs[-1].target_q, startup)
    state.q[:12] = startup
    for _ in range(10):
        node._tick()
    assert node._prepose_pass >= 10
    node._manager.commit.assert_not_called()
    response = node._on_confirm_engage(None, SimpleNamespace())
    assert response.success
    node._tick()
    node._manager.commit.assert_called_once()
    node._manager.run.assert_called_once()
    node._release_band.assert_not_called()

    # Independent upstream first observation: zero prior action/history and encoder factors.
    frame = np.r_[np.zeros(3), 1., np.zeros(3), [0., 0., -1.],
                  state.q - np.r_[defaults, np.zeros(15)], np.zeros(27), np.zeros(12)]
    with torch.no_grad():
        latent = policy._encoder(torch.zeros(1, 21)).numpy()[0]
        observation = np.r_[frame, np.zeros(152), np.zeros(16), latent].astype(np.float32)
        raw = policy._policy(torch.from_numpy(observation)[None]).numpy()[0]
    expected = np.clip(defaults + .25 * raw, policy._lower_limit, policy._upper_limit)
    np.testing.assert_allclose(policy._action, raw, atol=3e-6)
    np.testing.assert_allclose(outputs[-1].target_q, expected, atol=1e-6)
    delta = np.abs(outputs[-1].target_q - startup)
    assert np.all(delta <= np.abs(defaults - startup) + .25 * np.abs(raw) + 1e-6)
    assert np.max(delta) == pytest.approx(.3390802, abs=3e-6)
    first_policy_publish = len(events) - 1
    node._tick()
    assert events.index("release") > first_policy_publish
    node._band_released = True
    node._tick()
    assert not node._awaiting_band_release
    node._manager.reset_active.assert_not_called()
    assert len(policy._obs_history) == 3
    node._trip_heartbeat.assert_not_called()


@pytest.mark.parametrize("joint", range(12))
def test_fame_settling_checks_every_measured_joint_not_nominal(joint):
    node, policy, state, _, _ = fame_startup_fixture()
    state.q[:12] = policy.startup_lower
    for _ in range(100):
        node._tick()
    state.q[joint] += .16
    for _ in range(10):
        node._tick()
    assert node._prepose_pass == 0
    node._manager.commit.assert_not_called()
    state.q[:12] = policy.startup_lower
    for _ in range(10):
        node._tick()
    assert node._prepose_pass == 10
    node._manager.commit.assert_not_called()


def test_fame_early_confirmation_cannot_skip_upstream_ramp():
    node, policy, state, _, _ = fame_startup_fixture()
    state.q[:12] = policy.startup_lower
    node._engage_confirmed = True
    for _ in range(99):
        node._tick()
    node._manager.commit.assert_not_called()
    assert node._prepose_pass == 0
    for _ in range(10):
        node._tick()
    node._manager.commit.assert_called_once()


def test_fame_initial_joint_limits_rejected_before_any_mock_output():
    node, _, state, outputs, _ = fame_startup_fixture()
    state.q[0] = 1.
    node._tick()
    assert outputs == []
    node._trip_heartbeat.assert_called_once_with("FAME startup joints outside configured limits")
    node._manager.commit.assert_not_called()
    node._release_band.assert_not_called()


@pytest.mark.parametrize("cls,name", [(WalkPolicy, "walk"), (FamePolicy, "fame"),
                                      (AlmiPolicy, "almi"), VARIANTS[1]])
def test_other_policies_keep_nominal_cosine_prepose(cls, name):
    node, _, state, outputs, _ = fame_startup_fixture()
    policy = load(cls, name)
    node._standing_policy = policy if name == "almi_manip_stand" else None
    node._manager = PolicyManager({name: policy}, log=lambda msg: None)
    node._manager.request(name)
    node._prepose_ramp_ticks = 100
    state.q[:12] = policy.nominal_lower
    state.q[3] += .2
    initial = state.q[:12].copy()
    for tick in range(5):
        node._tick()
        alpha = .5 * (1 - np.cos(np.pi * tick / 100))
        np.testing.assert_allclose(outputs[-1].target_q,
                                   initial * (1 - alpha) + policy.nominal_lower * alpha)
    assert node._manager.is_idle()


@pytest.mark.parametrize("key,value", [
    ("lower_start_pos", [0.] * 11), ("lower_start_pos", [float("nan")] * 12),
    ("lower_start_pos", [100.] * 12), ("startup_seconds", 0),
])
def test_fame_invalid_startup_metadata_fails_before_model_load(key, value, tmp_path, monkeypatch):
    root = POLICIES / "fame_finetune"
    cfg = yaml.safe_load((root / "fame_finetune.yaml").read_text())
    for path_key in ("policy_path", "encoder_path"):
        cfg[path_key] = str(root / cfg[path_key])
    cfg[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(torch.jit, "load", lambda *a, **kw: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="startup"):
        FameFinetunePolicy(str(path))
