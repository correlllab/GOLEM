'''Stdout lines written by a thread of their own, so stdout never stops the relay'''

import os
import sys
import json
import threading
from collections import deque

# lines waiting for stdout; beyond this, new lines are dropped and counted
MAX_PENDING_LINES = 10000
# how long the exit waits for stdout to take the last lines
EXIT_DRAIN_SECONDS = 1.0


class Console:
    '''
    Queue lines for stdout and write them from a daemon thread: write_line()
    never blocks or raises, whether stdout is closed (a dead ros2 launch), hung
    up (a closed terminal) or stalled (a reader that stopped reading)
    '''
    def __init__(self) -> None:
        self._lines: deque[str] = deque()
        # lines queued and not yet written, including the one being written
        self._pending = 0
        self._dropped = 0
        self._broken = False
        self._changed = threading.Condition()
        self._writer: threading.Thread | None = None

    def write_line(self, line: str) -> None:
        '''Queue line (without its newline) for stdout'''
        with self._changed:
            if self._pending >= MAX_PENDING_LINES:
                self._dropped += 1
                return
            if self._writer is None:
                self._writer = threading.Thread(
                    target=self._run, name='console_writer', daemon=True
                )
                self._writer.start()
            self._lines.append(line)
            self._pending += 1
            self._changed.notify_all()

    def drain(self, timeout: float) -> bool:
        '''Wait up to timeout for every queued line to be written; False if stdout is stalled'''
        with self._changed:
            return self._changed.wait_for(lambda: self._pending == 0, timeout)

    def _run(self) -> None:
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._lines)
                line = self._lines.popleft()
                dropped, self._dropped = self._dropped, 0
            if dropped:
                report = {'event': 'stdout_lines_dropped', 'count': dropped}
                self._write(json.dumps(report, sort_keys=True))
            self._write(line)
            with self._changed:
                self._pending -= 1
                self._changed.notify_all()

    def _write(self, line: str) -> None:
        '''Write one line, once stdout fails drop it and every later one'''
        if self._broken:
            return
        try:
            # one write per line: print() writes the newline separately, so
            # another thread's print() could land inside the line
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
        except Exception:  # EPIPE, EIO, a closed file: stdout is gone for good
            self._broken = True
            _discard_stdout()


def _discard_stdout() -> None:
    '''
    Point the stdout file descriptor at /dev/null (Python's recipe for a broken
    pipe), so neither a print() elsewhere nor the exit flush fails on it
    '''
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except Exception:  # no file descriptor (e.g. a test's capture): nothing to point
        pass


# the process has one stdout, so it has one console
_CONSOLE = Console()


def write_line(line: str) -> None:
    '''Queue line (without its newline) for stdout; never blocks or raises'''
    _CONSOLE.write_line(line)


def drain(timeout: float) -> bool:
    '''Wait up to timeout for every queued line to be written; False if stdout is stalled'''
    return _CONSOLE.drain(timeout)


def finish(code: int) -> int:
    '''
    Return code for the process to exit with, once the last lines are out; if
    stdout is stalled, exit with it right here instead, since the interpreter's
    own exit flush would wait on stdout forever
    '''
    if not drain(EXIT_DRAIN_SECONDS):
        os._exit(code)
    return code
