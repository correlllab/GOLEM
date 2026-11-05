"""
Turn a segmented object cloud into ordered grasp candidates for /skill/grasp.

A grasp runs in three stages and this package owns the middle one:

  perception  (skills/grasp.py)   box -> segmented object cloud, pelvis frame
  synthesis   (this package)      cloud -> Candidates, in execution order
  execution   (skills/grasp.py)   approach the first reachable candidate, then
                                  either the open-loop contact move or the
                                  closed-loop visual servo (goal.visual_servo),
                                  then close

Every method takes the same SynthesisInput and returns Candidates in the same
gripper-base convention (common.py), and the executor drives every method's
output the same way, so swapping methods changes the grasp and nothing else.
The execution mode is independent of the method.

A goal runs `top_down` when goal.top_down is set, else the method named by the
skills node's `grasp_method` parameter (default DEFAULT_METHOD), e.g.
`ros2 param set /golem_skills grasp_method antipodal`.

A method is a module with `synthesize(inp) -> (Candidates, None) | (None,
reason)`, registered in METHODS.
"""

from dataclasses import dataclass
from typing import Callable

from . import antipodal, centroid, graspgenx, heuristic, top_down
from .common import Candidates, SynthesisInput


@dataclass(frozen=True)
class GraspMethod:
    """Name a synthesis function and how the executor approaches its grasps."""

    name: str
    # SynthesisInput -> (Candidates, None) | (None, reason)
    synthesize: Callable
    # True: approach from the arm's joint-space staging config (a repeatable
    # posture above the workspace) instead of 'prep', and reset to it between
    # failed candidates.
    staged: bool = False


METHODS = {m.name: m for m in (
    GraspMethod('heuristic', heuristic.synthesize),
    GraspMethod('top_down', top_down.synthesize, staged=True),
    GraspMethod('centroid', centroid.synthesize),
    GraspMethod('antipodal', antipodal.synthesize),
    GraspMethod('graspgenx', graspgenx.synthesize_raw),
    GraspMethod('graspgenx_filtered', graspgenx.synthesize_filtered),
)}

DEFAULT_METHOD = 'heuristic'


def method_name(top_down, configured):
    """Return the method a goal runs: `top_down` if the goal asks for it, else `configured`."""
    return 'top_down' if top_down else str(configured).strip()


__all__ = ['Candidates', 'DEFAULT_METHOD', 'GraspMethod', 'METHODS',
           'SynthesisInput', 'method_name']
