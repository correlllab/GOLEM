"""Check the existing object-only reset without creating a ROS node."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np


class ObjectResetTests(unittest.TestCase):
    def test_contact_measurements_are_read_only_and_identify_workcell(self):
        tree = ast.parse(Path(__file__).with_name('measurement_bridge.py').read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_init_experiment', '_publish_ground_truth', '_on_reset_objects')]
        harness = ast.ClassDef(name='ContactHarness', bases=[], keywords=[], body=methods,
                               decorator_list=[])
        namespace = {'mujoco': mujoco, 'np': np, 'json': json,
                     'String': SimpleNamespace, 'Trigger': object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[])),
                     'measurement_bridge.py', 'exec'), namespace)
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="table" pos="0 0 .5">
            <geom name="table_top" type="box" size=".3 .3 .1"/>
          </body>
          <body name="robot0_pelvis" pos="0 0 .65"><freejoint/>
            <geom name="robot0_hand" type="sphere" size=".1" mass="1"/>
          </body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        bridge = namespace['ContactHarness']()
        bridge.env = SimpleNamespace(
            sim=SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data)),
            robots=[SimpleNamespace(robot_model=SimpleNamespace(naming_prefix='robot0_'))])
        messages = []
        bridge.create_publisher = lambda *args: SimpleNamespace(publish=messages.append)
        bridge.create_service = lambda *args: None
        bridge.get_logger = lambda: SimpleNamespace(warn=lambda *args: None)
        bridge._init_experiment()
        before = (data.qpos.copy(), data.qvel.copy(), data.time)
        bridge._publish_ground_truth()
        contacts = json.loads(messages[-1].data)['robot_environment_contacts']
        self.assertTrue(contacts)
        contact = contacts[0]
        self.assertEqual(set(contact['geoms']), {'table_top', 'robot0_hand'})
        self.assertLess(contact['distance_m'], 0)
        self.assertGreater(contact['force_contact_N'][0], 0)
        self.assertEqual(len(contact['torque_contact_Nm']), 3)
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        self.assertEqual(data.time, before[2])

    def test_reset_preserves_robot_and_simulation_clock(self):
        tree = ast.parse(Path(__file__).with_name('measurement_bridge.py').read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_init_experiment', '_reset_objects', '_on_reset_objects')]
        harness = ast.ClassDef(name='ResetHarness', bases=[], keywords=[], body=methods,
                               decorator_list=[])
        namespace = {'mujoco': mujoco, 'String': object, 'Trigger': object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[])),
                     'measurement_bridge.py', 'exec'), namespace)
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="robot0_pelvis" pos="0 0 1"><freejoint/>
            <geom type="box" size=".1 .1 .1" mass="10"/>
            <body name="robot0_leg" pos="0 0 -.2"><joint type="hinge"/>
              <geom type="capsule" size=".02 .1" mass="1"/></body>
          </body>
          <body name="source_peg" pos=".5 .2 .9"><freejoint/>
            <geom type="cylinder" size=".006 .025" mass=".007"/>
          </body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        bridge = namespace['ResetHarness']()
        bridge.env = SimpleNamespace(
            sim=SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data)),
            robots=[SimpleNamespace(robot_model=SimpleNamespace(naming_prefix='robot0_'))])
        bridge.create_publisher = lambda *args: None
        services = []
        bridge.create_service = lambda _, name, callback: services.append(name)
        bridge.get_logger = lambda: SimpleNamespace(warn=lambda *args: None)
        bridge._init_experiment()
        self.assertEqual(services, ['/sim/reset_objects'])
        self.assertEqual(len(bridge._reset_joints), 1)
        qadr, nq, dadr, nv = bridge._reset_joints[0]
        data.qpos[0:3] = [.1, -.1, 1.2]
        data.qpos[qadr - 1] = .4
        data.qvel[:dadr] = .3
        data.qpos[qadr:qadr + 3] += [.2, .1, .3]
        data.qvel[dadr:dadr + nv] = 1.2
        data.time = 17.0
        robot_q, robot_dq = data.qpos[:qadr].copy(), data.qvel[:dadr].copy()
        bridge._reset_objects()
        np.testing.assert_array_equal(data.qpos[:qadr], robot_q)
        np.testing.assert_array_equal(data.qvel[:dadr], robot_dq)
        np.testing.assert_array_equal(data.qpos[qadr:qadr + nq], model.qpos0[qadr:qadr + nq])
        np.testing.assert_array_equal(data.qvel[dadr:dadr + nv], np.zeros(nv))
        self.assertEqual(data.time, 17.0)


if __name__ == '__main__':
    unittest.main()
