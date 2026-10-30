"""Policy registry + idle/active state.

The manager holds the lower-body policies and at most one *active* policy.
``active_name`` is ``None`` when **idle**: the robot is held by the elastic band
and no policy drives the legs. A requested policy becomes active once the node
releases the band and calls ``commit``, which ``reset()``s it for a clean
free-standing warm-up. The band itself is owned by the node, not here.
"""

from __future__ import annotations

from .policy import LegCommand, Policy, RobotState


class PolicyManager:
    def __init__(self, policies: dict[str, Policy], log=print):
        self._policies = policies
        self._active: str | None = None     # None = idle (band-held, no policy)
        self._desired: str | None = None
        self._log = log

    # -- introspection -------------------------------------------------------
    @property
    def active_name(self) -> str | None:
        return self._active

    @property
    def desired_name(self) -> str | None:
        return self._desired

    def is_idle(self) -> bool:
        return self._active is None

    def is_pending(self) -> bool:
        return self._desired is not None and self._desired != self._active

    def names(self) -> list[str]:
        return list(self._policies)

    def has(self, name: str) -> bool:
        return name in self._policies

    def desired_policy(self) -> Policy | None:
        """The pending (requested but not yet committed) policy, or None if idle."""
        return self._policies.get(self._desired) if self._desired is not None else None

    # -- control -------------------------------------------------------------
    def request(self, name: str) -> tuple[bool, str]:
        """Request a policy. Returns (accepted, message). It is committed later,
        after band release."""
        if name not in self._policies:
            return False, f"unknown policy {name!r}; have {self.names()}"
        if name == self._active and not self.is_pending():
            return True, f"{name!r} already active"
        if self.is_pending() and name == self._desired:
            return True, f"{name!r} already requested"
        self._desired = name
        self._log(f"[policy] requested activate (idle->{name})")
        return True, f"requested {name!r}"

    def commit(self, state: RobotState) -> str:
        """Activate the desired policy now (reset for a clean warm-up). Used for
        the first activation right after band release."""
        incoming = self._policies[self._desired]
        incoming.reset(state)
        old = self._active
        self._active = self._desired
        self._log(f"[policy] committed {old!r} -> {self._active!r}")
        return self._active

    def reset_active(self, state: RobotState) -> None:
        if self._active is not None:
            self._policies[self._active].reset(state)

    def run(self, state: RobotState) -> LegCommand:
        return self._policies[self._active].compute(state)
