"""Check check_lowerbody.py's scoring against synthetic runs that should pass or fail."""
import math
import unittest
from check_lowerbody import score_run


def run_data():
    """Return a passing run: rows, phases and policies of 5 s balance, 3 s walk and 3 s stop."""
    rows=[]
    for step in range(551):
        t=step*.02
        forward=max(0., min(.45, (t-5.)*.15))
        joints=[0.]*27
        if 5. <= t <= 8.:
            joints[3]=.4+.1*math.sin((t-5.)*2*math.pi)
            joints[9]=.4-.1*math.sin((t-5.)*2*math.pi)
        rows.append(dict(sim_time=t, position=[forward,0.,1.], quaternion=[1.,0.,0.,0.], joint_positions=joints, support_active=False))
    return rows, dict(balance=(0.,5.),walk=(5.,8.),stop=(8.,11.)), [(0.,'almi')]


class LowerbodyScoringTests(unittest.TestCase):
    """Score a passing run and variants of it with one fault each."""

    def test_successful_walk_on_almi(self):
        """A run that balances, walks 0.45 m and stops, on ALMI throughout, passes."""
        report=score_run(*run_data())
        self.assertAlmostEqual(report['forward_displacement'],.45)
    def test_supported_robot_cannot_pass_balance(self):
        """A support active in a scored phase fails."""
        rows,phases,policies=run_data();rows[10]['support_active']=True
        with self.assertRaisesRegex(AssertionError,'support'):
            score_run(rows,phases,policies)
    def test_motion_without_bilateral_leg_articulation_fails(self):
        """Forward motion with no knee movement fails."""
        rows,phases,policies=run_data()
        for row in rows: row['joint_positions']=[0.]*27
        with self.assertRaisesRegex(AssertionError,'knees'):
            score_run(rows,phases,policies)
    def test_fall_fails_even_with_forward_progress(self):
        """A base that drops to 0.5 m fails despite forward progress."""
        rows,phases,policies=run_data();rows[320]['position'][2]=.5
        with self.assertRaisesRegex(AssertionError,'collapsed'):
            score_run(rows,phases,policies)
    def test_excess_tilt_fails(self):
        """A base tilted past the limit fails."""
        rows,phases,policies=run_data();rows[320]['quaternion']=[math.cos(.4),math.sin(.4),0.,0.]
        with self.assertRaisesRegex(AssertionError,'tilt'):
            score_run(rows,phases,policies)
    def test_stationary_command_echo_is_not_walking(self):
        """Walking-policy state without forward displacement fails."""
        rows,phases,policies=run_data()
        for row in rows: row['position'][0]=0.
        with self.assertRaisesRegex(AssertionError,'Forward displacement'):
            score_run(rows,phases,policies)
    def test_policy_must_stay_active(self):
        """A run whose policy stops (idle, a fault) in a scored phase fails."""
        rows,phases,policies=run_data();policies.append((9.,'idle'))
        with self.assertRaisesRegex(AssertionError,'expected active policy almi'):
            score_run(rows,phases,policies)
    def test_expected_policy_is_selectable(self):
        """With `expected_policy=walk`, WALK throughout passes and ALMI fails."""
        rows,phases,_=run_data()
        score_run(rows,phases,[(0.,'walk')],expected_policy='walk')
        with self.assertRaisesRegex(AssertionError,'expected active policy walk'):
            score_run(rows,phases,[(0.,'almi')],expected_policy='walk')
    def test_missing_telemetry_cannot_hide_fall(self):
        """A one-second telemetry gap fails as stale."""
        rows,phases,policies=run_data();rows=[r for r in rows if not 6. < r['sim_time'] < 7.]
        with self.assertRaisesRegex(AssertionError,'stale'):
            score_run(rows,phases,policies)
    def test_forward_scoring_follows_initial_heading(self):
        """Forward displacement is measured along the robot's initial heading."""
        rows,phases,policies=run_data()
        for r in rows:
            r['position'][1],r['position'][0]=r['position'][0],0.
            r['quaternion']=[math.sqrt(.5),0.,0.,math.sqrt(.5)]
        self.assertAlmostEqual(score_run(rows,phases,policies)['forward_displacement'],.45)
    def test_missing_support_telemetry_is_error(self):
        """A row without `support_active` fails."""
        rows,phases,policies=run_data();del rows[10]['support_active']
        with self.assertRaisesRegex(AssertionError,'support_active'):
            score_run(rows,phases,policies)


if __name__=='__main__': unittest.main()
