"""Actual native ROS services/actions with synthetic physics, without a GPU."""
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from src.python.sensors.magpie_bridge import MagpieBridge
from src.python.sensors.ros_sensor_bridge import RosSensorBridge

sensors = RosSensorBridge(SimpleNamespace(sensors={}))
bridge = MagpieBridge()
probe = None
try:
    launcher = Path('/home/code/CL_isaaclab_sim/src/python/sensors/magpie_launch.sh')
    probe = subprocess.Popen(['bash', str(launcher), '--script', '/home/code/tests/isaaclab/probe_magpie_services.py'])
    started = time.monotonic()
    while probe.poll() is None and time.monotonic() - started < 40:
        commands = bridge.get_commands()
        states = {side: dict(position=values['aperture'], force=0., finger_positions=[values['aperture']]*2,
                            is_moving=False, contact_detected=False) for side, values in commands.items()}
        bridge.publish(time.monotonic() - started, states)
        bridge.acknowledge(commands)
        time.sleep(.02)
    assert probe.poll() == 0, f'native ROS probe failed/timed out: {probe.poll()}'
finally:
    if probe is not None and probe.poll() is None:
        probe.terminate()
        probe.wait(timeout=5)
    bridge.close()
    sensors.close()
