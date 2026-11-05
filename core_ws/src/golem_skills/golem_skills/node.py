#!/usr/bin/env python3
"""
Serve the /skill/<name> actions (both robots) from one node.

SkillsNode serves the 12 Skill* actions of custom_ros_messages listed in
base.SKILL_ACTIONS. SkillsBase (base.py) holds the clients, camera caches,
perception pipeline and motion primitives; each skill's execute callback is a
mixin under skills/.

Skills run inside action-server execute callbacks and block there on inner
service and action futures, so the node spins on a MultiThreadedExecutor with
one reentrant callback group: the other threads keep the futures completing.
"""

from functools import partial

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from .base import SkillsBase, SKILL_ACTIONS
from model_server.model_logging import ModelLogger, declare_logging_params
from .skills import (
    OpenDoorSkill, CloseDoorSkill, OpenLidSkill, CloseLidSkill, NavigateSkill,
    GraspSkill, PickPlaceSkill, PressSkill, SlideRackSkill, TurnSkill,
    FrontierExploreSkill,
)


def _describe_msg(msg):
    """
    Flatten an action goal's primitive fields into a JSON-friendly dict.

    Lists and nested messages are summarized (item count, type name), not dumped.
    """
    out = {}
    try:
        for fname in msg.get_fields_and_field_types():
            v = getattr(msg, fname)
            if isinstance(v, (bool, int, float, str)):
                out[fname] = v
            elif isinstance(v, (list, tuple, bytes)):
                out[fname] = f'[{len(v)} items]'
            else:
                out[fname] = type(v).__name__
    except Exception as e:
        out['_error'] = str(e)
    return out


class SkillsNode(SkillsBase, OpenDoorSkill, CloseDoorSkill, OpenLidSkill,
                 CloseLidSkill, NavigateSkill, GraspSkill, PickPlaceSkill,
                 PressSkill, SlideRackSkill, TurnSkill, FrontierExploreSkill):
    """Serve every /skill/<name> action on top of SkillsBase."""

    def __init__(self):
        """Declare the logging toggles and create one action server per SKILL_ACTIONS entry."""
        super().__init__()

        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'skills', 'golem_skills', __file__,
                                  log=log, visualize=viz, clear=clear)
        self._frontier_init()

        # turn_lever and twist_knob share _exec_turn, bound to their action
        # type, label and motion phase
        executors = {
            'open_door':  self._exec_open_door,
            'close_door': self._exec_close_door,
            'open_lid':   self._exec_open_lid,
            'close_lid':  self._exec_close_lid,
            'navigate':   self._exec_navigate,
            'grasp':      self._exec_grasp,
            'pick_place': self._exec_pick_place,
            'press':      self._exec_press,
            'slide_rack': self._exec_slide_rack,
            'turn_lever': partial(self._exec_turn, SKILL_ACTIONS['turn_lever'][0],
                                  'turn_lever', 'turn'),
            'twist_knob': partial(self._exec_turn, SKILL_ACTIONS['twist_knob'][0],
                                  'twist_knob', 'twist'),
            'frontier_explore': self._exec_frontier_explore,
        }
        # frontier_explore validates its fields and runs one goal at a time
        goal_callbacks = {
            'frontier_explore': self._on_frontier_goal,
        }
        self.skill_servers = {
            name: ActionServer(
                self, SKILL_ACTIONS[name][0], SKILL_ACTIONS[name][1],
                execute_callback=self._safe_exec(
                    name, SKILL_ACTIONS[name][0], executors[name]),
                cancel_callback=self._on_skill_cancel,
                callback_group=self._cb_group,
                **({'goal_callback': goal_callbacks[name]}
                   if name in goal_callbacks else {}),
            )
            for name in SKILL_ACTIONS
        }

        self.get_logger().info(
            f'golem_skills ready: serving {sorted(self.skill_servers)} on /skill/<name>')

    def _safe_exec(self, label, action_type, fn):
        """
        Wrap a skill execute callback so an escaping exception aborts the goal.

        The wrapper logs the call (ModelLogger) and the run's telemetry, and an
        exception (including a stub's NotImplementedError) aborts the goal with
        'internal error: ...' instead of leaving it stuck in EXECUTING.
        """
        def _cb(goal_handle):
            """Run `fn` on the goal and record its outcome."""
            rec = self.logger.start()
            rec.set(skill=label, request=_describe_msg(goal_handle.request))
            self._log_head_frame(rec, f'{label}_start')
            # None unless recording is on, the skill is recorded and no
            # recorder is active (pick_place's inner grasp shares the outer one)
            run_rec = self._start_run_telemetry(label, goal_handle)
            try:
                result = fn(goal_handle)
                success = bool(getattr(result, 'success', False))
                message = getattr(result, 'message', '')
                rec.finish(success=success, message=message)
                self._finish_run_telemetry(run_rec, success=success, message=message)
                return result
            except Exception as e:
                self.get_logger().error(f'[{label}] internal error: {e}')
                result = action_type.Result()
                result.success = False
                result.message = f'internal error: {e}'
                goal_handle.abort()
                rec.finish(success=False, message=result.message)
                self._finish_run_telemetry(run_rec, success=False,
                                           message=result.message)
                return result
        return _cb

    def _log_head_frame(self, rec, tag):
        """Save the latest head-camera frame captioned with `tag`, when visualization is on."""
        if not self.logger.visualize:
            return
        img = self.latest_image()
        if img is None:
            return
        try:
            import cv2
            import numpy as np
            buf = np.frombuffer(bytes(img.data), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            cv2.putText(bgr, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            rec.save_overlay(tag, bgr)
        except Exception as e:
            self.get_logger().warn(f'head-frame viz failed: {e}')


def main():
    """Spin SkillsNode on an 8-thread MultiThreadedExecutor until interrupted."""
    rclpy.init()
    node = SkillsNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
