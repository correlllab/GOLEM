#!/usr/bin/env python3
"""
Log each model call's inputs, outputs and overlays to disk, behind two ROS parameters.

Each model node (gemini, sam, graspgen, yolo, and golem_skills' skills node)
declares the parameters with declare_logging_params:

    enable_logging        save raw inputs/outputs and a per-call metadata record
    enable_visualization  save annotated overlay images
    clear_logs            wipe this model's log directory at startup (default true)

Output goes to a package-relative tree, so it persists on the host through the
core_ws bind mount and survives container restarts:

    <package>/logs/<model_name>/
        calls.jsonl                  one JSON line per call (when enabled)
        <id>_<tag>.{png,npy,txt}     per-call artifacts

<id> is "<YYYYmmdd_HHMMSS_mmm>_<seq>", so artifacts sort chronologically and
group by call. MODEL_LOG_DIR overrides the output root.

golem_skills imports this module from model_server (its exec_depend).
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime

import numpy as np


def _log_root(pkg_name, anchor_file):
    """
    Return the "<package>/logs" directory of `pkg_name`; ModelLogger appends the model's subdir.

    In order: MODEL_LOG_DIR; the bind-mounted source tree
    (/home/code/core_ws/src/<pkg>), so output persists on the host rather than
    in a build/ or install/ tree a clean rebuild wipes; the directory holding
    package.xml above `anchor_file`; the working directory.
    """
    override = os.environ.get('MODEL_LOG_DIR')
    if override:
        return override
    src = os.path.join('/home/code/core_ws/src', pkg_name)
    if os.path.isdir(src):
        return os.path.join(src, 'logs')
    d = os.path.dirname(os.path.realpath(anchor_file))
    for _ in range(8):
        if os.path.isfile(os.path.join(d, 'package.xml')):
            return os.path.join(d, 'logs')
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.join(os.getcwd(), 'logs')


def _imwrite(path, img, rgb):
    """Write an image with OpenCV; `img` is BGR unless `rgb`."""
    import cv2
    img = np.asarray(img)
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


class _Call:
    """
    Collect one model call's artifacts and metadata record.

    Every save is gated by the parent logger's toggles, so node code calls
    them unconditionally.
    """

    def __init__(self, logger, seq):
        """Start the call's latency clock and name its artifacts by time and `seq`."""
        self._logger = logger
        self._t0 = time.monotonic()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        self.id = f'{stamp}_{seq:04d}'
        self.meta = {'id': self.id, 'model': logger.model_name}

    def path(self, tag, ext):
        """Return the absolute path of this call's `tag` artifact; the caller writes it."""
        return os.path.join(self._logger.dir, f'{self.id}_{tag}.{ext}')

    def set(self, **fields):
        """Merge `fields` into this call's metadata record; return self."""
        self.meta.update(fields)
        return self

    def save_image(self, tag, img, rgb=False):
        """Save an input/output image as PNG when enable_logging is on."""
        if self._logger.log:
            self._safe(lambda: _imwrite(self.path(tag, 'png'), img, rgb), tag)

    def save_array(self, tag, arr):
        """Save an array as .npy when enable_logging is on."""
        if self._logger.log:
            self._safe(lambda: np.save(self.path(tag, 'npy'), np.asarray(arr)), tag)

    def save_text(self, tag, text):
        """Save text as .txt when enable_logging is on."""
        if self._logger.log:
            def _w():
                """Write the text file."""
                with open(self.path(tag, 'txt'), 'w') as f:
                    f.write('' if text is None else str(text))
            self._safe(_w, tag)

    def save_overlay(self, tag, img, rgb=False):
        """Save an annotated image as PNG when enable_visualization is on."""
        if self._logger.visualize:
            self._safe(lambda: _imwrite(self.path(tag, 'png'), img, rgb), tag)

    def _safe(self, fn, tag):
        """Run the write `fn`; a failure is logged, never raised (logging never fails a call)."""
        try:
            fn()
        except Exception as e:
            self._logger._warn(f'failed to write {tag}: {e}')

    def finish(self, success=None, message=None):
        """Append the metadata record, with latency_ms filled in, to calls.jsonl; call once."""
        if success is not None:
            self.meta['success'] = bool(success)
        if message is not None:
            self.meta['message'] = str(message)
        self.meta['latency_ms'] = round((time.monotonic() - self._t0) * 1000.0, 1)
        try:
            line = json.dumps(self.meta, default=str)
            with self._logger._lock:
                with open(self._logger._meta_path, 'a') as f:
                    f.write(line + '\n')
        except Exception as e:
            self._logger._warn(f'failed to append metadata: {e}')


class _NullCall:
    """Accept and ignore every _Call method, so node code needs no branch when output is off."""

    id = ''

    def path(self, *a, **k):
        """Return os.devnull."""
        return os.devnull

    def set(self, **_):
        """Ignore the fields; return self."""
        return self

    def save_image(self, *a, **k):
        """Do nothing."""
        pass

    def save_array(self, *a, **k):
        """Do nothing."""
        pass

    def save_text(self, *a, **k):
        """Do nothing."""
        pass

    def save_overlay(self, *a, **k):
        """Do nothing."""
        pass

    def finish(self, *a, **k):
        """Do nothing."""
        pass


class ModelLogger:
    """
    Own one model's output directory and its logging and visualization toggles.

    Construct from the declare_logging_params values; call start() once per
    request to get that call's artifact sink.
    """

    def __init__(self, node, model_name, pkg_name, anchor_file,
                 log=False, visualize=False, clear=False):
        """Resolve the output directory, clear it when asked, and create it when output is on."""
        self._node = node
        self.model_name = model_name
        self.log = bool(log)
        self.visualize = bool(visualize)
        self.dir = os.path.join(_log_root(pkg_name, anchor_file), model_name)
        self._meta_path = os.path.join(self.dir, 'calls.jsonl')
        self._lock = threading.Lock()
        self._seq = 0
        # cleared whatever the toggles say, so stale artifacts go even with output off
        if clear:
            self._clear_dir()
        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
                self._info(f'logging={self.log} visualization={self.visualize} '
                           f'-> {self.dir}')
            except OSError as e:
                self._warn(f'cannot create {self.dir}: {e}; disabling output')
                self.log = self.visualize = False

    def _clear_dir(self):
        """Remove this model's log directory, best-effort."""
        try:
            if os.path.isdir(self.dir):
                shutil.rmtree(self.dir)
                self._info(f'cleared previous logs in {self.dir}')
        except OSError as e:
            self._warn(f'could not clear {self.dir}: {e}')

    @property
    def enabled(self):
        """Return True when logging or visualization is on."""
        return self.log or self.visualize

    def start(self):
        """Begin a call; return its sink, a no-op one when both toggles are off."""
        if not self.enabled:
            return _NullCall()
        with self._lock:
            self._seq += 1
            seq = self._seq
        return _Call(self, seq)

    def _info(self, msg):
        """Log `msg` at info level, tagged with the model name."""
        self._node.get_logger().info(f'[{self.model_name} log] {msg}')

    def _warn(self, msg):
        """Log `msg` at warn level, tagged with the model name."""
        self._node.get_logger().warn(f'[{self.model_name} log] {msg}')


def declare_logging_params(node, *, log_default=False, viz_default=False,
                           clear_default=True):
    """Declare enable_logging, enable_visualization and clear_logs on `node`; return the values."""
    log = node.declare_parameter('enable_logging', log_default).value
    viz = node.declare_parameter('enable_visualization', viz_default).value
    clear = node.declare_parameter('clear_logs', clear_default).value
    return bool(log), bool(viz), bool(clear)
