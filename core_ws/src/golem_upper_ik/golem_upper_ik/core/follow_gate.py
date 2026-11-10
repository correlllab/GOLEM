"""Hold the upper-body command near the measured pose until golem_safety relays it."""

import numpy as np


class FollowGate:
    """Keep each gated joint's command within tolerance (rad) of its measured q until open().

    SAFETY: before golem_safety relays the controller's commands, nothing executes them: it
    streams zero gains until it engages the input. An open-loop move such as the start-up
    move to home would then run away from the measured pose, and the robot would jump once
    the commands are relayed. With the gate a command leads its measured joint by at most
    tolerance, before the relay starts and at it.

    The gate opens when the safety layer is seen relaying the commands (relays_commands),
    never on how the measured joints move: it takes a relayed command for an executed one and
    does not check that the joints follow. Once open it never moves a command again, so a
    held joint resists a push or a tipping torso instead of following it. Tolerance 0 turns
    the gate off (open from the start).
    """

    def __init__(self, joint_ids, tolerance):
        """Gate body slots joint_ids at tolerance, rad; ValueError if negative or not finite."""
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError('follow tolerance must be zero (off) or positive')
        self._ids = np.asarray(list(joint_ids), dtype=np.int64)
        self._tolerance = tolerance
        self.following = tolerance == 0.0

    @property
    def joint_ids(self):
        """Return a copy of the gated body slots."""
        return self._ids.copy()

    @property
    def tolerance(self):
        """Return the tolerance, rad."""
        return self._tolerance

    def open(self):
        """Let every command through from now on: the safety layer relays them."""
        self.following = True

    def limit(self, q_cmd, q_measured):
        """Return q_cmd with each gated joint held within tolerance of q_measured."""
        q_cmd = np.array(q_cmd, dtype=np.float64)
        if self.following:
            return q_cmd
        measured = np.asarray(q_measured, dtype=np.float64)[self._ids]
        q_cmd[self._ids] = np.clip(
            q_cmd[self._ids], measured - self._tolerance, measured + self._tolerance
        )
        return q_cmd


def relays_commands(sample_q, sample_kp, q_cmd, joint_ids, tolerance):
    """Return whether a golem_safety output sample carries q_cmd on every given joint.

    Each joint's q must be within tolerance (rad) of q_cmd's, with kp > 0.
    """
    ids = np.asarray(list(joint_ids), dtype=np.int64)
    # the safety layer's estop and zero-gain commands and a default LowCmd have kp 0: never ours
    if not np.all(np.asarray(sample_kp, dtype=np.float64)[ids] > 0.0):
        return False
    sample_q = np.asarray(sample_q, dtype=np.float64)[ids]
    error = np.abs(sample_q - np.asarray(q_cmd, dtype=np.float64)[ids])
    # a non-finite sample never matches
    return bool(np.all(error <= tolerance))
