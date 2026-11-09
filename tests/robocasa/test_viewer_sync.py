"""Check the shared viewer thread (viewer_sync.py) against a fake viewer handle, without a display."""
import contextlib
import io
import threading
import time
import unittest

import mujoco
import numpy as np

import viewer_sync

# A free body and a hinged link.
XML = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
    <body name="arm" pos="1 0 1"><joint name="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass="1"/></body>
  </worldbody>
</mujoco>
"""


class FakeScene:
    """Stand in for the viewer's user scene."""

    ngeom = 5


class FakeHandle:
    """Stand in for a passive viewer handle: counts syncs, optionally writes a mouse force."""

    def __init__(self, perturb=None):
        """Run until close(); perturb (n_bodies x 6) is written into the viewer's data on sync."""
        self.running = True
        self.syncs = 0
        self.viewer_data = None
        self.perturb = perturb
        self.user_scn = FakeScene()

    def is_running(self):
        """Return whether the viewer window is open."""
        return self.running

    def sync(self):
        """Count the sync and apply the mouse force to the viewer's data."""
        self.syncs += 1
        if self.perturb is not None:
            self.viewer_data.xfrc_applied[:] = self.perturb

    def close(self):
        """Close the fake window."""
        self.running = False


def wait_for(condition, timeout=2.0):
    """Return whether condition() became true within timeout (s)."""
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class ViewerSyncTests(unittest.TestCase):
    """Check ViewerSync's copy, perturbation, shutdown and overlay behaviour."""

    def setUp(self):
        """Build the model, the simulation's data (moved off its start) and the viewer's own data."""
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.viewer_data = mujoco.MjData(self.model)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.lock = threading.Lock()

    def make(self, handle, draw=None, hz=200.0):
        """Return a ViewerSync over the fake handle, registered for cleanup."""
        handle.viewer_data = self.viewer_data
        sync = viewer_sync.ViewerSync(self.model, self.data, handle, self.lock, self.viewer_data,
                                      draw=draw, hz=hz)
        self.addCleanup(sync.stop)
        return sync

    def test_viewer_sync_copies_state_under_the_lock(self):
        """The viewer's state follows the simulation's, and a held lock blocks the copy."""
        handle = FakeHandle()
        sync = self.make(handle)
        self.assertFalse(np.allclose(self.viewer_data.qpos, self.data.qpos))
        with self.lock:
            sync.start()
            time.sleep(0.1)
            self.assertEqual(handle.syncs, 0)
            self.assertFalse(np.allclose(self.viewer_data.qpos, self.data.qpos))
        self.assertTrue(wait_for(lambda: handle.syncs > 0))
        with self.lock:
            np.testing.assert_array_equal(self.viewer_data.qpos, self.data.qpos)
            np.testing.assert_array_equal(self.viewer_data.qvel, self.data.qvel)
            self.assertEqual(self.viewer_data.time, self.data.time)

    def test_viewer_sync_returns_the_perturbation(self):
        """The mouse force the viewer wrote during sync() reaches the simulation's xfrc_applied."""
        force = np.zeros((self.model.nbody, 6))
        force[1, :3] = (1.0, -2.0, 3.0)
        handle = FakeHandle(perturb=force)
        sync = self.make(handle)
        sync.start()
        self.assertTrue(wait_for(lambda: handle.syncs > 0))
        self.assertTrue(wait_for(lambda: self._perturbed(sync)))
        np.testing.assert_array_equal(self.data.xfrc_applied, force)

    def _perturbed(self, sync):
        """Apply the perturbation under the lock and return whether it landed."""
        with self.lock:
            sync.apply_perturbation()
            return bool(np.any(self.data.xfrc_applied))

    def test_viewer_sync_stops_when_handle_closes(self):
        """The thread ends within two periods of the window closing (plus 0.1 s of scheduling slack), and stop() returns."""
        hz = 50.0
        handle = FakeHandle()
        sync = self.make(handle, hz=hz)
        sync.start()
        self.assertTrue(wait_for(lambda: handle.syncs > 0))
        handle.close()
        thread = sync._thread
        thread.join(timeout=2.0 / hz + 0.1)
        self.assertFalse(thread.is_alive())
        sync.stop()

    def test_draw_runs_each_frame(self):
        """draw is called once per sync with the handle and the viewer's MjData."""
        calls = []
        handle = FakeHandle()
        sync = self.make(handle, draw=lambda h, d: calls.append((h, d, handle.syncs)))
        sync.start()
        self.assertTrue(wait_for(lambda: handle.syncs >= 3))
        sync.stop()
        self.assertEqual(len(calls), handle.syncs)
        self.assertTrue(all(h is handle and d is self.viewer_data for h, d, _ in calls))
        # draw runs before each sync, so the n-th call sees n - 1 syncs.
        self.assertEqual([n for _, _, n in calls], list(range(len(calls))))

    def test_without_draw_the_overlay_is_cleared(self):
        """With no draw callback the viewer's user scene is emptied each frame."""
        handle = FakeHandle()
        sync = self.make(handle)
        sync.start()
        self.assertTrue(wait_for(lambda: handle.syncs > 0))
        self.assertEqual(handle.user_scn.ngeom, 0)

    def test_an_error_ends_the_thread_and_is_logged(self):
        """A sync() that raises ends the thread, with the error and its traceback in the log."""
        handle = FakeHandle()

        def broken_sync():
            """Fail as a closed GL context would."""
            raise RuntimeError("sync failed")

        handle.sync = broken_sync
        sync = self.make(handle)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            sync.start()
            sync._thread.join(timeout=2.0)
        self.assertFalse(sync._thread.is_alive())
        self.assertIn("[viewer_sync] the viewer's sync thread stopped", log.getvalue())
        self.assertIn("RuntimeError: sync failed", log.getvalue())


if __name__ == "__main__":
    unittest.main()
