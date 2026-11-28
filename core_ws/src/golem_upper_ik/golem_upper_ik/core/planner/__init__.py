"""Plan collision-free arm paths with OMPL: the planner golem_ompl's planner_server runs."""
from golem_upper_ik.core.planner.reduced_joint_planner import (
    PlannerConfig,
    ReducedJointPlanner,
)


__all__ = [
    'PlannerConfig',
    'ReducedJointPlanner',
]
