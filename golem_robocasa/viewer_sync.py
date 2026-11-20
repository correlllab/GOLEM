"""Run the passive MuJoCo viewer on its own thread, from its own MjData, for both RoboCasa sims.

The viewer's handle.sync() takes ~11 ms because it waits for the render thread,
so a physics loop that calls it every step runs far below real time. Here a
60 Hz thread copies the simulation's display state into a separate MjData the
viewer draws, syncs, and hands the mouse perturbation back for the physics loop
to apply. The thread also shows the measured real-time factor in the viewer's
top-left corner (RtfMeter). open_viewer and close_viewer start and end the
viewer and the thread for both sim loops.
"""
import collections
import math
import threading
import time
import traceback

import mujoco
import numpy as np

from mujoco_env import draw_band

# Hz, wall. The sync thread's rate.
VIEWER_HZ = 60.0
# s, wall. The real-time factor shown is the sim time advanced over this window of wall time.
RTF_WINDOW_S = 1.0
# s, wall. How often the shown real-time factor is refreshed, so the digits can be read.
RTF_TEXT_PERIOD_S = 0.25


class RtfMeter:
    """Measure the real-time factor, sim seconds per wall second, over RTF_WINDOW_S of wall time."""

    def __init__(self, window=RTF_WINDOW_S):
        """Start empty; window is in seconds of wall time."""
        self._window = window
        self._samples = collections.deque()

    def add(self, wall, sim):
        """Record sim time `sim` (s) seen at wall time `wall` (s).

        A sim time that went back (MuJoCo reset the scene) restarts the window.
        """
        if self._samples and sim < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((wall, sim))
        while len(self._samples) > 2 and wall - self._samples[1][0] >= self._window:
            self._samples.popleft()

    def rtf(self):
        """Return the real-time factor over the window, or None before two samples span wall time."""
        if len(self._samples) < 2:
            return None
        (w0, s0), (w1, s1) = self._samples[0], self._samples[-1]
        return (s1 - s0) / (w1 - w0) if w1 > w0 else None


def rtf_text(rtf, sim_time, target=None):
    """Return the viewer's (label, value) text columns: the real-time factor, its target and sim time."""
    value = '-' if rtf is None else f'{rtf:.2f}'
    if target is not None:
        value += ' (target max)' if math.isinf(target) else f' (target {target:g})'
    return 'real-time factor\nsim time', f'{value}\n{sim_time:.1f} s'


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


def open_viewer(model, data, lock, band, body, tag, sim_rate=None, key_callbacks=()):
    """Launch the passive viewer on its own copy of data, framed on body, and start its sync thread.

    lock is the physics loop's lock (ViewerSync). band, the sim's ElasticBand
    or None, is drawn while enabled and toggled by SPACE. Collision geoms are
    hidden. Each of key_callbacks also gets every key the viewer passes on.
    A framing failure is logged under [tag]. sim_rate, the target real-time
    factor (sim_common.sim_rate_from_env), is shown beside the measured one.
    Returns (handle, view_sync); close them with close_viewer.
    """
    import mujoco.viewer

    import sim_common

    viewer_data = mujoco.MjData(model)
    with lock:
        copy_display_state(viewer_data, data)
    mujoco.mj_forward(model, viewer_data)
    # SPACE in the viewer toggles the band (ElasticBand.key_callback).
    callbacks = ((band.key_callback,) if band is not None else ()) + tuple(key_callbacks)
    if callbacks:
        def on_key(keycode):
            """Pass a viewer key to every callback."""
            for callback in callbacks:
                callback(keycode)
        handle = mujoco.viewer.launch_passive(model, viewer_data, key_callback=on_key)
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
                           sim_rate=sim_rate)
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
                 sim_rate=None):
        """Set up the sync of data into viewer_data, drawn by the passive viewer handle.

        lock is the threading.Lock the physics loop holds around mj_step and
        after_step (and around apply_perturbation); the copy takes it too.
        viewer_data is the MjData the viewer was launched with. draw(handle,
        viewer_data), if given, runs once per frame before the sync to put the
        overlay in handle.user_scn; without it the overlay is cleared.
        Each frame also feeds data.time to an RtfMeter, whose real-time factor, with
        sim_rate (the target, if given), is set as the viewer's text every RTF_TEXT_PERIOD_S.
        """
        self._model = model
        self._data = data
        self._handle = handle
        self._lock = lock
        self._viewer_data = viewer_data
        self._draw = draw
        self._period = 1.0 / hz
        self._perturb = np.zeros_like(data.xfrc_applied)
        self._sim_rate = sim_rate
        self._rtf = RtfMeter()
        self._next_text = 0.0
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
        """Copy the state, show the real-time factor, draw the overlay, sync and keep the mouse force."""
        viewer_data = self._viewer_data
        with self._lock:
            copy_display_state(viewer_data, self._data)
        now = time.perf_counter()
        self._rtf.add(now, viewer_data.time)
        if now >= self._next_text:
            self._next_text = now + RTF_TEXT_PERIOD_S
            label, value = rtf_text(self._rtf.rtf(), viewer_data.time, self._sim_rate)
            self._handle.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, value))
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
