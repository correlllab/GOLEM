"""Synthetic truth cases for lower-body physical acceptance thresholds."""
import copy
import math
import unittest
from check_lowerbody import score_run


def run_data():
    rows=[]
    for step in range(551):
        t=step*.02
        forward=max(0., min(.45, (t-5.)*.15))
        joints=[0.]*27
        if 5. <= t <= 8.:
            joints[3]=.4+.1*math.sin((t-5.)*2*math.pi)
            joints[9]=.4-.1*math.sin((t-5.)*2*math.pi)
        rows.append(dict(sim_time=t, position=[forward,0.,1.], quaternion=[1.,0.,0.,0.], joint_positions=joints, support_active=False))
    return rows, dict(balance=(0.,5.),walk=(5.,8.),stop=(8.,11.)), [(0.,'almi'),(5.1,'walk'),(8.5,'almi')]


class LowerbodyScoringTests(unittest.TestCase):
    def test_successful_walk_and_return_to_almi(self):
        report=score_run(*run_data())
        self.assertAlmostEqual(report['forward_displacement'],.45)
    def test_supported_robot_cannot_pass_balance(self):
        rows,phases,policies=run_data();rows[10]['support_active']=True
        with self.assertRaisesRegex(AssertionError,'support'):
            score_run(rows,phases,policies)
    def test_motion_without_bilateral_leg_articulation_fails(self):
        rows,phases,policies=run_data()
        for row in rows: row['joint_positions']=[0.]*27
        with self.assertRaisesRegex(AssertionError,'knees'):
            score_run(rows,phases,policies)
    def test_fall_fails_even_with_forward_progress(self):
        rows,phases,policies=run_data();rows[320]['position'][2]=.5
        with self.assertRaisesRegex(AssertionError,'collapsed'):
            score_run(rows,phases,policies)
    def test_excess_tilt_fails(self):
        rows,phases,policies=run_data();rows[320]['quaternion']=[math.cos(.4),math.sin(.4),0.,0.]
        with self.assertRaisesRegex(AssertionError,'tilt'):
            score_run(rows,phases,policies)
    def test_stationary_command_echo_is_not_walking(self):
        rows,phases,policies=run_data()
        for row in rows: row['position'][0]=0.
        with self.assertRaisesRegex(AssertionError,'Forward displacement'):
            score_run(rows,phases,policies)
    def test_must_return_to_almi_after_stop(self):
        rows,phases,policies=run_data();policies.pop()
        with self.assertRaisesRegex(AssertionError,'expected active policy almi'):
            score_run(rows,phases,policies)
    def test_direct_almi_locomotion_is_selectable(self):
        rows,phases,_=run_data()
        score_run(rows,phases,[(0.,'almi')],expected_walk_policy='almi')
    def test_missing_telemetry_cannot_hide_fall(self):
        rows,phases,policies=run_data();rows=[r for r in rows if not 6. < r['sim_time'] < 7.]
        with self.assertRaisesRegex(AssertionError,'stale'):
            score_run(rows,phases,policies)
    def test_forward_scoring_follows_initial_heading(self):
        rows,phases,policies=run_data()
        for r in rows:
            r['position'][1],r['position'][0]=r['position'][0],0.
            r['quaternion']=[math.sqrt(.5),0.,0.,math.sqrt(.5)]
        self.assertAlmostEqual(score_run(rows,phases,policies)['forward_displacement'],.45)
    def test_missing_support_telemetry_is_error(self):
        rows,phases,policies=run_data();del rows[10]['support_active']
        with self.assertRaisesRegex(AssertionError,'support_active'):
            score_run(rows,phases,policies)


if __name__=='__main__': unittest.main()
