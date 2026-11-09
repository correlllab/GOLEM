"""Run the passive MuJoCo viewer on its own thread, from its own MjData, for both RoboCasa sims.

The viewer's handle.sync() takes ~11 ms because it waits for the render thread,
so a physics loop that calls it every step runs far below real time. Here a
60 Hz thread copies the simulation's display state into a separate MjData the
viewer draws, syncs, and hands the mouse perturbation back for the physics loop
to apply. open_viewer and close_viewer start and end the viewer and the
thread for both sim loops.
"""
import threading
import time
import traceback

import mujoco
import numpy as np

from mujoco_env import draw_band

# Hz, wall. The sync thread's rate.
VIEWER_HZ = 60.0


def copy_display_state(dst, src):
    """Copy the state the viewer draws from src to dst; the caller holds the physics lock."""
    dst.time = src.time
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.act[:] = src.act
    dst.mocap_pos[:] = src.mocap_pos
    dst.mocap_quat[:] = src.mocap_quat


def band_overlay(band, body_id):
    """Return a draw callback that shows the elastic band on body_id while it is enabled."""
    def draw(handle, viewer_data):
        """Draw the band from its anchor to the body, or clear the overlay once released."""
        if band.enabled:
            draw_band(handle, band.point, viewer_data.xpos[body_id])
        else:
            handle.user_scn.ngeom = 0
    return draw


def open_viewer(model, data, lock, band, body, tag, timings=None):
    """Launch the passive viewer on its own copy of data, framed on body, and start its sync thread.

    lock is the physics loop's lock (ViewerSync). band, the sim's ElasticBand
    or None, is drawn while enabled and toggled by SPACE. Collision geoms are
    hidden. A framing failure is logged under [tag]. Returns (handle,
    view_sync); close them with close_viewer.
    """
    import mujoco.viewer

    import sim_common

    viewer_data = mujoco.MjData(model)
    with lock:
        copy_display_state(viewer_data, data)
    mujoco.mj_forward(model, viewer_data)
    if band is not None:
        # SPACE in the viewer toggles the band (ElasticBand.key_callback).
        handle = mujoco.viewer.launch_passive(model, viewer_data, key_callback=band.key_callback)
    else:
        handle = mujoco.viewer.launch_passive(model, viewer_data)
    handle.opt.geomgroup[0] = 0   # hide collision geoms
    if body >= 0:
        try:
            with handle.lock():
                sim_common.frame_viewer_on_robot(handle, viewer_data, body)
        except Exception as e:
            print(f"[{tag}] viewer camera framing skipped: {e}")
    view_sync = ViewerSync(model, data, handle, lock, viewer_data,
                           draw=band_overlay(band, body) if band is not None else None,
                           timings=timings)
    view_sync.start()
    return handle, view_sync


def close_viewer(handle, view_sync):
    """Stop the sync thread, then close the viewer; either may be None."""
    if view_sync is not None:
        view_sync.stop()
    if handle is not None:
        handle.close()


class ViewerSync:
    """Refresh the viewer's MjData from the simulation's and sync the viewer at hz."""

    def __init__(self, model, data, handle, lock, viewer_data, draw=None, hz=VIEWER_HZ,
                 timings=None):
        """Set up the sync of data into viewer_data, drawn by the passive viewer handle.

        lock is the threading.Lock the physics loop holds around mj_step and
        after_step (and around apply_perturbation); the copy takes it too.
        viewer_data is the MjData the viewer was launched with. draw(handle,
        viewer_data), if given, runs once per frame before the sync to put the
        overlay in handle.user_scn; without it the overlay is cleared. timings,
        a sim_timing.LoopTimings, records each frame's wall time as stage "viewer".
        """
        self._model = model
        self._data = data
        self._handle = handle
        self._lock = lock
        self._viewer_data = viewer_data
        self._draw = draw
        self._period = 1.0 / hz
        self._timings = timings
        self._perturb = np.zeros_like(data.xfrc_applied)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        """Start the sync thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="viewer_sync")
        self._thread.start()

    def stop(self):
        """Ask the thread to end and wait for it; a stop before start() does nothing."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def apply_perturbation(self):
        """Copy the viewer's mouse perturbation into data.xfrc_applied; the caller holds the lock."""
        self._data.xfrc_applied[:] = self._perturb

    def _run(self):
        """Sync frames at the configured rate until stop() or the viewer closes.

        An exception ends the thread, logged with its traceback; the physics loop goes on with
        the viewer no longer refreshed.
        """
        try:
            while not self._stop.is_set() and self._handle.is_running():
                frame_start = time.perf_counter()
                self._frame()
                self._stop.wait(max(0.0, self._period - (time.perf_counter() - frame_start)))
        except Exception:
            print("[viewer_sync] the viewer's sync thread stopped; the viewer no longer "
                  "refreshes:\n" + traceback.format_exc(), flush=True)

    def _frame(self):
        """Copy the state, draw the overlay, sync the viewer and keep its mouse force."""
        viewer_data = self._viewer_data
        mark = self._timings.start() if self._timings is not None else 0.0
        with self._lock:
            copy_display_state(viewer_data, self._data)
        mujoco.mj_forward(self._model, viewer_data)
        # Zeroed so that after sync() it holds only the mouse perturbation.
        viewer_data.xfrc_applied[:] = 0.0
        if self._draw is not None:
            self._draw(self._handle, viewer_data)
        else:
            self._handle.user_scn.ngeom = 0
        self._handle.sync()
        with self._lock:
            self._perturb[:] = viewer_data.xfrc_applied
        if self._timings is not None:
            self._timings.stop("viewer", mark)
