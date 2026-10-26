import unittest
from sim_common import select_robot

class SelectRobotTests(unittest.TestCase):
    def test_default_is_h12(self):
        self.assertEqual(select_robot([], env={}), 'h12')
    def test_env_selects_g1_case_insensitively(self):
        self.assertEqual(select_robot([], env={'GOLEM_ROBOT': 'G1'}), 'g1')
    def test_flag_beats_env(self):
        self.assertEqual(select_robot(['--h12'], env={'GOLEM_ROBOT': 'g1'}), 'h12')
        self.assertEqual(select_robot(['--task', 'X', '--g1'], env={}), 'g1')
    def test_empty_env_is_default(self):
        self.assertEqual(select_robot([], env={'GOLEM_ROBOT': ''}), 'h12')
    def test_unknown_env_exits(self):
        with self.assertRaises(SystemExit):
            select_robot([], env={'GOLEM_ROBOT': 'h1'})
