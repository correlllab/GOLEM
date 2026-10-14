"""Grasp synthesis for /skill/grasp: perception output -> ordered grasps.

A grasp run has three stages, and this package owns only the middle one:

  perception  (skills/grasp.py)   box -> segmented object cloud, pelvis frame
  synthesis   (this package)      cloud -> Candidates, in execution order
  execution   (skills/grasp.py)   approach the first reachable candidate,
                                  then either the open-loop contact move or
                                  the closed-loop visual servo
                                  (goal.visual_servo), then close

Every method receives the same SynthesisInput and returns Candidates in the
same gripper-base convention, and the executor drives every method's output
the same way, so swapping methods changes the grasp and nothing else.
Execution mode is orthogonal: any method can be run open-loop or with the
visual servo.

Choosing a method: goal.top_down selects `top_down`; otherwise the skills
node's `grasp_method` parameter does (default DEFAULT_METHOD), e.g.
`ros2 param set /h12_skills grasp_method antipodal`.

To add a method: write a module with `synthesize(inp) -> (Candidates, None)
| (None, reason)` and register it in METHODS.
"""

from dataclasses import dataclass
from typing import Callable

from . import antipodal, centroid, graspgenx, heuristic, top_down
from .common import Candidates, SynthesisInput


@dataclass(frozen=True)
class GraspMethod:
    name: str
    synthesize: Callable    # SynthesisInput -> (Candidates, None) | (None, reason)
    # Approach from the arm's joint-space staging config (a repeatable posture
    # above the workspace) instead of 'prep'; also the reset config between
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
    """Name of the method a goal runs: goal.top_down forces `top_down`,
    otherwise the configured `grasp_method` parameter value."""
    return 'top_down' if top_down else str(configured).strip()


__all__ = ['Candidates', 'DEFAULT_METHOD', 'GraspMethod', 'METHODS',
           'SynthesisInput', 'method_name']
