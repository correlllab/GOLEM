"""Check that the G1 sim's loop paces to GOLEM_SIM_RATE through the shared pacer (sim_timing.RealtimePacer).

Each test runs g1_mujoco.sim_loop headless in the Blank task, sensors off, in a subprocess
(it starts ROS and Unitree DDS), with the pacer replaced by a subclass that records how it was
built and counts its waits. Run it in the robocasa image with golem_robocasa on PYTHONPATH;
tests/robocasa/g1/run_in_container.sh runs it with no network on ROS_DOMAIN_ID 97.
"""
import json
import os
import subprocess
import sys
import textwrap

DURATION = 0.05  # s of sim time each loop runs

LOOP = textwrap.dedent("""
    import json
    import sim_timing
    from g1 import g1_mujoco

    pacers = []

    class RecordedPacer(sim_timing.RealtimePacer):
        def __init__(self, timestep, rate=1.0, *args, **kwargs):
            super().__init__(timestep, rate, *args, **kwargs)
            pacers.append({"timestep": timestep, "rate": rate, "waits": 0})

        def wait(self):
            pacers[-1]["waits"] += 1
            return super().wait()

    g1_mujoco.RealtimePacer = RecordedPacer
    g1_mujoco.sim_loop("Blank", viewer=False, duration=%r)
    print("PACERS " + json.dumps(pacers))
""" % DURATION)


def _run(rate):
    """Run the G1 loop with GOLEM_SIM_RATE=rate, sensors off; return the CompletedProcess."""
    env = dict(os.environ, GOLEM_SIM_RATE=rate, GOLEM_LIDAR="0", GOLEM_CAMERAS="0")
    r = subprocess.run([sys.executable, "-c", LOOP], capture_output=True, text=True, env=env, timeout=300)
    print(textwrap.indent((r.stdout + r.stderr).strip()[-1500:], "  "))
    return r


def _pacers(r):
    """Return the pacers the loop built, from its PACERS line."""
    assert r.returncode == 0, r.returncode
    line = next(line for line in r.stdout.splitlines() if line.startswith("PACERS "))
    return json.loads(line[len("PACERS "):])


def test_rate_two_paces_every_step_at_twice_real_time():
    """GOLEM_SIM_RATE=2 builds one pacer at rate 2 on the physics step, waited on once per step, and is logged."""
    r = _run("2")
    (pacer,) = _pacers(r)
    assert pacer["rate"] == 2.0, pacer
    assert pacer["waits"] == round(DURATION / pacer["timestep"]), pacer
    assert "[g1_mujoco] sim rate: 2x real time (GOLEM_SIM_RATE)" in r.stdout


def test_default_is_real_time():
    """GOLEM_SIM_RATE empty paces to 1x real time."""
    r = _run("")
    (pacer,) = _pacers(r)
    assert pacer["rate"] == 1.0, pacer
    assert "[g1_mujoco] sim rate: 1x real time (GOLEM_SIM_RATE)" in r.stdout


def test_max_is_unpaced():
    """GOLEM_SIM_RATE=max builds the unpaced pacer (rate infinity)."""
    r = _run("max")
    (pacer,) = _pacers(r)
    assert pacer["rate"] == float("inf"), pacer
    assert "[g1_mujoco] sim rate: unpaced" in r.stdout


def test_bad_rate_stops_before_the_scene_is_built():
    """GOLEM_SIM_RATE=abc stops the sim, naming the knob and the value, before it builds the scene."""
    r = _run("abc")
    assert r.returncode != 0
    assert "GOLEM_SIM_RATE" in r.stderr and "'abc'" in r.stderr, r.stderr
    assert "RoboCasa task" not in r.stdout


if __name__ == "__main__":
    # Each file runs as a script (run_in_container.sh), so it runs its own test_* functions.
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
