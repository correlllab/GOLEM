"""Run the Isaac sim with --elastic_band and check the band holds, releases and re-holds the robot (GPU).

run_elastic_band.sh runs this in place of sim_main.py. It wraps the env's step to record the
torso, and a thread calls /elastic_band/toggle over ROS, from a context of its own, at two
sim times. Nothing drives the motors, so the robot hangs limp on the band, collapses when it
is released, and is pulled back to the anchor, from wherever it lay, when it is re-enabled:
a force applied in the torso's frame instead of the world's would not bring it back.
Each check prints PASS or FAIL and the last line PASSED or FAILED, which run_elastic_band.sh reads:
Kit's shutdown sets the exit status.
"""
import math
from pathlib import Path
import runpy
import sys
import threading
import time

ENTRY = Path('/home/code/CL_isaaclab_sim/sim_main.py')
RELEASE_T = 40.0    # s of sim time: the first toggle, which releases the band; the leg node holds under it up to 34 s
REHOLD_T = 44.0      # s of sim time: the second toggle, which enables it again
END_T = 50.0         # s of sim time the run lasts
STEP_DT = 0.01       # s, the env step sim_main.py runs at (--max_steps counts these)
HOLD_RADIUS = 0.10   # m, how far from the anchor the held torso may hang
HOLD_TILT = 20.0     # deg, the held torso's largest tilt
FALLEN_DROP = 0.30   # m, how far below the anchor the released torso must fall


def tilt_deg(quaternion):
    """Return the angle (deg) between a body's z axis and world up, from its WXYZ quaternion."""
    w, x, y, z = quaternion
    return math.degrees(math.acos(max(-1., min(1., 1. - 2. * (x * x + y * y)))))


class Run:
    """Hold what the wrapped env records and what the toggle thread hears."""

    def __init__(self):
        """Start with no samples and no replies."""
        self.rows = []        # (sim time s, torso position m world, tilt deg, band enabled)
        self.anchor = None    # m, world: the band's anchor
        self.sim_time = 0.
        self.replies = []     # the Trigger replies' (success, message), in call order
        self.error = None

    def wrap(self, make):
        """Return gym.make wrapped to record the torso after every env step."""
        def observed_make(*args, **kwargs):
            """Make the Isaac env and wrap its step."""
            wrapper = make(*args, **kwargs)
            env = wrapper.unwrapped
            step = env.step
            robot = env.scene['robot']
            torso = robot.data.body_names.index('torso_link')

            def observed_step(*a, **kw):
                """Step the env and record the torso and the band's state."""
                result = step(*a, **kw)
                band = env.elastic_band
                self.anchor = band.anchor[0].tolist()
                self.sim_time = env._sim_step_counter * env.physics_dt
                self.rows.append((self.sim_time, robot.data.body_pos_w[0, torso].tolist(),
                                  tilt_deg(robot.data.body_quat_w[0, torso].tolist()), band.enabled))
                # Report on the last step: Kit's shutdown ends the process once sim_main.py's loop exits.
                if len(self.rows) == round(END_T / STEP_DT):
                    self.report()
                return result

            env.step = observed_step
            return wrapper
        return observed_make

    def toggle_at(self, times):
        """Call /elastic_band/toggle once sim time passes each of times; record the replies."""
        try:
            # Isaac's bundled ROS Python, as the sim's own bridge finds it.
            from src.python.sensors.ros_sensor_bridge import bootstrap_ros
            rclpy = bootstrap_ros()
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from std_srvs.srv import Trigger
            context = Context()
            rclpy.init(context=context)
            node = rclpy.create_node('elastic_band_check', context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            client = node.create_client(Trigger, '/elastic_band/toggle')
            for when in times:
                while self.sim_time < when:
                    time.sleep(0.01)
                if not client.wait_for_service(timeout_sec=10.0):
                    raise RuntimeError('/elastic_band/toggle is not served')
                future = client.call_async(Trigger.Request())
                executor.spin_until_future_complete(future, timeout_sec=10.0)
                if not future.done():
                    raise RuntimeError('/elastic_band/toggle did not answer within 10 s')
                self.replies.append((future.result().success, future.result().message))
        except Exception as error:   # reported by the checks
            self.error = error

    def distance(self, row):
        """Return how far (m) a row's torso is from the anchor."""
        return math.dist(row[1], self.anchor)

    def checks(self):
        """Return [(passed, text)] for the run."""
        held = [r for r in self.rows if 3.0 <= r[0] < RELEASE_T]
        fallen = [r for r in self.rows if REHOLD_T - 0.5 <= r[0] < REHOLD_T]
        reheld = [r for r in self.rows if r[0] >= END_T - 1.0]
        results = [(self.error is None and self.replies == [(True, 'disabled'), (True, 'enabled')],
                    f"/elastic_band/toggle answered {self.replies} (expected disabled, then enabled); error {self.error}")]
        if not (held and fallen and reheld):
            return results + [(False, f'the run recorded {len(self.rows)} samples, too few for every phase')]
        results.append((max(map(self.distance, held)) <= HOLD_RADIUS and max(r[2] for r in held) <= HOLD_TILT,
                        f'held: torso at most {max(map(self.distance, held)):.3f} m from the anchor (limit {HOLD_RADIUS}), '
                        f'tilt at most {max(r[2] for r in held):.1f} deg (limit {HOLD_TILT})'))
        drop = self.anchor[2] - max(r[1][2] for r in fallen)
        results.append((drop >= FALLEN_DROP and not any(r[3] for r in fallen),
                        f'released: torso {drop:.3f} m below the anchor before the second toggle (at least {FALLEN_DROP}), '
                        f'tilt {fallen[-1][2]:.1f} deg'))
        results.append((max(map(self.distance, reheld)) <= HOLD_RADIUS and all(r[3] for r in reheld),
                        f're-held from the fallen pose: torso at most {max(map(self.distance, reheld)):.3f} m from '
                        f'the anchor in the last second (limit {HOLD_RADIUS}), tilt {reheld[-1][2]:.1f} deg'))
        return results


    def report(self):
        """Print the torso every 0.5 s, each check as PASS or FAIL, then PASSED or FAILED with the count."""
        for row in self.rows[::50]:
            print(f'[check] t={row[0]:5.2f} s torso {[round(v, 3) for v in row[1]]} m, '
                  f'{self.distance(row):.3f} m from the anchor, tilt {row[2]:5.1f} deg, band {row[3]}', flush=True)
        results = self.checks()
        for passed, text in results:
            print(f"{'PASS' if passed else 'FAIL'} {text}", flush=True)
        print(f"{'PASSED' if all(p for p, _ in results) else 'FAILED'}: "
              f"{sum(p for p, _ in results)}/{len(results)} checks", flush=True)


def main():
    """Run sim_main.py with the band and toggle it twice; the last step prints the checks."""
    run = Run()
    sys.path.insert(0, str(ENTRY.parent))
    import gymnasium as gym
    gym.make = run.wrap(gym.make)
    threading.Thread(target=run.toggle_at, args=((RELEASE_T, REHOLD_T),), daemon=True).start()
    sys.argv = [str(ENTRY), '--elastic_band', '--max_steps', str(round(END_T / STEP_DT))] + sys.argv[1:]
    runpy.run_path(str(ENTRY), run_name='__main__')


if __name__ == '__main__':
    main()
