"""Ask golem_ompl's planner_server for a planned arm path: PlanClient.

The controller plans through /plan_arm_trajectory (custom_ros_messages/PlanArmTrajectory),
which golem_ompl's planner_server serves; golem_ompl imports this package, so nothing here
imports it. A call waits for the service to appear (the server builds its models at start)
for up to SERVICE_WAIT, then for the reply, which ends only with the reply, the server going
away, or shutdown(). Calls from several threads take turns.
"""
import threading
import time

from custom_ros_messages.srv import PlanArmTrajectory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

__all__ = ['PLAN_SERVICE', 'PlanClient', 'PlannerUnavailableError']

# the planner's service, golem_ompl's planner_server
PLAN_SERVICE = '/plan_arm_trajectory'
# steady s a call waits for the service to appear: the server serves only once its models
# are built
SERVICE_WAIT = 30.0
# steady s between checks while waiting
POLL_PERIOD = 0.01


class PlannerUnavailableError(RuntimeError):
    """Signal that the controller cannot plan: no self-collision model, or planning is off."""


class PlanClient:
    """Call /plan_arm_trajectory on a node (module doc)."""

    def __init__(self, node):
        """Create the service client on node, in a callback group of its own.

        The reply is taken in that group, so it arrives while the caller's own callback (a
        goal's) waits for it under a multi-threaded executor.
        """
        self._client = node.create_client(PlanArmTrajectory, PLAN_SERVICE,
                                          callback_group=MutuallyExclusiveCallbackGroup())
        self._shut_down = threading.Event()
        # one call at a time
        self._lock = threading.Lock()

    def call(self, request):
        """Return the PlanArmTrajectory response to request.

        RuntimeError when the service does not appear within SERVICE_WAIT, when it goes away
        before it replies, and ('planner shut down') once shutdown() has run.
        """
        with self._lock:
            deadline = time.monotonic() + SERVICE_WAIT
            while not self._client.service_is_ready():
                self._check_shut_down()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'the planner service {PLAN_SERVICE} did not appear '
                                       f"within {SERVICE_WAIT:.0f} s: is golem_ompl's "
                                       'planner_server running?')
                time.sleep(POLL_PERIOD)
            future = self._client.call_async(request)
            replied = threading.Event()
            future.add_done_callback(lambda _: replied.set())
            while not replied.wait(POLL_PERIOD):
                self._check_shut_down()
                if not self._client.service_is_ready():
                    raise RuntimeError(f'the planner service {PLAN_SERVICE} went away before '
                                       'it replied')
            return future.result()

    def _check_shut_down(self):
        """Raise RuntimeError ('planner shut down') once shutdown() has run."""
        if self._shut_down.is_set():
            raise RuntimeError('planner shut down')

    def shutdown(self):
        """Make a call waiting now, and every later call, raise 'planner shut down'."""
        self._shut_down.set()
