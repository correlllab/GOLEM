'''ROS 2 node running HOMIE's walking policy on the G1's legs'''

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Twist

from g1_locomotion.band import BandRelease
from g1_locomotion.config import init_channel_factory, load_config, resolve_config_path
from g1_locomotion.driver import LocomotionDriver
from g1_locomotion.locomotion import LocomotionController, load_policy

# where the ros container mounts this package's policy/ (deploy.onnx, g1.yaml, homie_policy.py)
DEFAULT_POLICY_DIR = '/home/code/g1_locomotion'
# how often the band release is checked (s)
BAND_PERIOD = 0.1


class LocomotionNode(Node):
    '''
    - /cmd_vel (geometry_msgs/Twist): linear.x, linear.y (m/s) and angular.z
      (rad/s) in the base frame; older than commands.timeout means stop
    - /locomotion/height (std_msgs/Float32): the pelvis height target, m
      (standing, 0.74, until one arrives)
    - /safety/heartbeat (std_msgs/Bool): golem_safety's. The node starts after
      the first true beat; a false beat, or none for heartbeat.timeout once
      the first has come, stops it for good
    - /lowerbody/confirm_engage (std_srvs/Trigger): the operator's go-ahead
      while the legs hold the default pose (startup.wait_for_confirm); refused
      at any other time
    - with band.release (the sim): /left_ee_pose (geometry_msgs/PoseStamped),
      which frame_task_server publishes once its start-up moves are done, and
      a client of the sim's /elastic_band/toggle (std_srvs/Trigger), called
      from a 10 Hz timer as BandRelease says: once the policy runs and a
      /left_ee_pose has come, or band.max_wait after the policy started
    The legs go to golem_safety's lower-body input over Unitree DDS; see
    LocomotionController for the startup and the faults.

    use_sim_time is always false, whatever the launch passes: the node times
    everything by rt/lowstate's tick and the monotonic clock, and sim time
    would only add a /clock subscription (500 Hz in the sim) in Python
    '''

    def __init__(self, config_name='g1.yaml'):
        no_sim_time = Parameter('use_sim_time', Parameter.Type.BOOL, False)
        super().__init__('locomotion_node', parameter_overrides=[no_sim_time])
        self.declare_parameter('policy_dir', DEFAULT_POLICY_DIR)
        policy_dir = self.get_parameter('policy_dir').value
        policy = load_policy(policy_dir)
        config_path = resolve_config_path(config_name)
        config = load_config(config_path, policy.config['cmd_range'])

        startup = config['startup']
        self.get_logger().info(
            f'Loaded {config_path} and the policy in {policy_dir}. Silent until golem_safety '
            f'beats on /safety/heartbeat, {config["topics"]["low_state"]} is steady and upright '
            f'and the arm controller commands on {config["topics"]["upper_cmd"]}; then the legs '
            f'move to the default pose over {startup["ramp_seconds"]:.1f} s'
            + (', hold it until /lowerbody/confirm_engage is called,'
               if startup['wait_for_confirm'] else '')
            + ' and the policy starts'
            + ('; the elastic band is released once the policy runs and /left_ee_pose has come'
               if config['band']['release'] else ''))

        domain_id = init_channel_factory(config)
        controller = LocomotionController(policy, config, log=self._log)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(Float32, '/locomotion/height', self._on_height, 10)
        self.create_subscription(Bool, '/safety/heartbeat', self._on_heartbeat, 10)
        self.create_service(Trigger, '/lowerbody/confirm_engage', self._on_confirm)
        self._driver = LocomotionDriver(controller, config['topics'], domain_id)

        self._band = BandRelease(config['band']['release'], config['band']['max_wait'])
        self._frame_task_ready = False
        self._pose_sub = self._band_client = self._band_timer = None
        if self._band.enabled:
            self._pose_sub = self.create_subscription(
                PoseStamped, '/left_ee_pose', self._on_ee_pose, 10)
            self._band_client = self.create_client(Trigger, '/elastic_band/toggle')
            self._band_timer = self.create_timer(BAND_PERIOD, self._on_band_timer)

    def _log(self, level, message):
        # one call per severity: rclpy refuses a change of severity at one call site
        if level == 'error':
            self.get_logger().error(message)
        elif level == 'warn':
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _on_cmd_vel(self, msg: Twist):
        self._driver.set_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_height(self, msg: Float32):
        self._driver.set_height(msg.data)

    def _on_heartbeat(self, msg: Bool):
        self._driver.on_heartbeat(msg.data)

    def _on_confirm(self, request, response):
        response.success, response.message = self._driver.confirm()
        self._log('info' if response.success else 'warn',
                  f'/lowerbody/confirm_engage: {response.message}')
        return response

    def _on_ee_pose(self, msg: PoseStamped):
        self._frame_task_ready = True

    def _on_band_timer(self):
        if self._frame_task_ready and self._pose_sub is not None:
            # one pose is all the release needs: stop taking frame_task_server's 100 Hz
            self.destroy_subscription(self._pose_sub)
            self._pose_sub = None
        missing = self._band.service_missing
        reason = self._band.poll(time.monotonic(), self._driver.walking_since,
                                 self._frame_task_ready, self._band_client.service_is_ready())
        if self._band.service_missing and not missing:
            self._log('warn', 'the elastic band is due for release, but /elastic_band/toggle '
                              'is not there yet: waiting for it')
        if reason is None:
            return
        self._log('info', f'releasing the elastic band: {reason}')
        self._band_client.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)

    def _on_band_toggled(self, future):
        try:
            response = future.result()
            failure = 'no response' if response is None else response.message
        except Exception as e:  # noqa: BLE001 - a failed call is sent again
            response, failure = None, repr(e)
        success = response is not None and response.success
        self._band.result(success)
        if success:
            self._log('info', f'elastic band released (/elastic_band/toggle: {response.message})')
            self._band_timer.cancel()
            if self._pose_sub is not None:
                self.destroy_subscription(self._pose_sub)
                self._pose_sub = None
        else:
            self._log('warn', f'/elastic_band/toggle failed ({failure}): calling it again')

    def destroy_node(self):
        # stop publishing; golem_safety keeps relaying the last leg command and
        # reports the quiet input
        self._driver.shutdown()
        return super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="HOMIE's walking policy on the G1's legs")
    parser.add_argument('--config', type=str, default='g1.yaml',
                        help='config name under config/ (g1.yaml or sim.yaml), or a path')
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = LocomotionNode(config_name=parsed_args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
