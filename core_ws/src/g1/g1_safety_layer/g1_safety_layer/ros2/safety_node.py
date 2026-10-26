'''ROS2 executable entrypoint for the G1 safety layer'''

import sys
import argparse
import threading
from typing import Any
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory

from g1_safety_layer.core import console
from g1_safety_layer.core.exit_signals import ExitRequest
from g1_safety_layer.core.safety_layer import SafetyLayer
from g1_safety_layer.core.config import dds_domain_id, init_channel_factory, load_config

PACKAGE_NAME = 'g1_safety_layer'

def _ensure_yaml_extension(config_path: Path) -> Path:
    '''Append .yaml when config path has no suffix'''
    if config_path.suffix:
        return config_path
    return config_path.with_suffix('.yaml')

def _resolve_config_path(config_name: str) -> Path:
    '''Resolve config from either bare name or config/<name> path'''
    config_path = _ensure_yaml_extension(Path(config_name))
    if config_path.is_absolute():
        return config_path
    share_dir = Path(get_package_share_directory(PACKAGE_NAME))
    if config_path.parts and config_path.parts[0] == 'config':
        return share_dir / config_path
    return share_dir / 'config' / config_path

class SafetyNode(Node):
    '''ROS2 node that manages SafetyLayer lifecycle'''
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        super().__init__('safety_node')
        init_channel_factory(config)
        self._safety_layer = SafetyLayer(config)
        self.failed = False
        # whether destroy_node() finished handing control back in time
        self.handed_back = False
        self._worker = threading.Thread(
            target=self._run_safety_layer,
            name='safety_layer_worker',
            daemon=True,
        )
        self._worker.start()
        self.get_logger().info(f'Loaded config: {config_path}')

    def _run_safety_layer(self) -> None:
        try:
            self._safety_layer.start()
        except Exception as exc:  # reported here; main() then exits non-zero
            self.failed = True
            self.get_logger().error(f'Safety layer exited with error: {exc}')

    def running(self) -> bool:
        '''Whether the safety layer is still running'''
        return self._worker.is_alive()

    def destroy_node(self, reason: str = 'Node destroyed') -> bool:
        '''Hand control back (see SafetyLayer.shutdown), then destroy the node'''
        self.handed_back = self._safety_layer.shutdown(reason)
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        return super().destroy_node()

def main(args: list[str] | None = None) -> int:
    '''Parse CLI args, start ROS2 node, spin until an exit signal; return the exit code'''
    parser = argparse.ArgumentParser(description='G1 low_cmd safety relay')
    parser.add_argument(
        '--config',
        type=str,
        default='arm_sdk.yaml',
        help='Config name or path, e.g. arm_sdk.yaml or config/split.yaml',
    )
    parsed_args, ros_args = parser.parse_known_args(args)
    config_path = _resolve_config_path(parsed_args.config)
    config = load_config(config_path)
    # before rclpy joins a DDS domain: a sim config refuses domain 0 here, and
    # rclpy joins the domain the Unitree channels use, also with ROS_DOMAIN_ID
    # unset (rclpy alone would then join domain 0)
    domain_id = dds_domain_id(config)

    # SIGINT, SIGTERM and SIGHUP hand control back before the process ends;
    # rclpy's own handlers would end spin() at once, skipping the arm_sdk
    # release ramp
    exit_request = ExitRequest()
    exit_request.install()
    rclpy.init(
        args=ros_args,
        domain_id=domain_id,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = SafetyNode(config, config_path)
    try:
        while not exit_request.is_set() and node.running():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if exit_request.is_set():
            reason = f'Exit requested ({exit_request.signal_name})'
        else:
            reason = 'Safety layer stopped'
        node.destroy_node(reason)
    if rclpy.ok():
        rclpy.shutdown()
    # without an exit signal the safety layer stopped by itself: a failure, as
    # is a hand-back that did not finish in time
    clean = exit_request.is_set() and not node.failed and node.handed_back
    return console.finish(0 if clean else 1)


if __name__ == '__main__':
    sys.exit(main())
