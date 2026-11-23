"""Check sim_common's robot selection and the command-line options both RoboCasa runners share."""
import argparse
import contextlib
import io
import unittest

from sim_common import add_common_args, add_robot_args, env_flag, parse_args, select_robot


class SelectRobotTests(unittest.TestCase):
    """Check select_robot: --h12/--g1, else GOLEM_ROBOT, else the H1-2."""

    def test_default_is_h12(self):
        """With no flag and no GOLEM_ROBOT the robot is the H1-2."""
        self.assertEqual(select_robot([], env={}), 'h12')
    def test_env_selects_g1_case_insensitively(self):
        """GOLEM_ROBOT=G1 selects the G1."""
        self.assertEqual(select_robot([], env={'GOLEM_ROBOT': 'G1'}), 'g1')
    def test_flag_beats_env(self):
        """A robot flag anywhere in the arguments wins over GOLEM_ROBOT."""
        self.assertEqual(select_robot(['--h12'], env={'GOLEM_ROBOT': 'g1'}), 'h12')
        self.assertEqual(select_robot(['--task', 'X', '--g1'], env={}), 'g1')
    def test_empty_env_is_default(self):
        """An empty GOLEM_ROBOT selects the H1-2."""
        self.assertEqual(select_robot([], env={'GOLEM_ROBOT': ''}), 'h12')
    def test_unknown_env_exits(self):
        """An unknown GOLEM_ROBOT exits."""
        with self.assertRaises(SystemExit):
            select_robot([], env={'GOLEM_ROBOT': 'h1'})
    def test_both_flags_exit(self):
        """Both robot flags exit with a message naming the choice."""
        for argv in (['--h12', '--g1'], ['--g1', '--task', 'X', '--h12']):
            with self.assertRaises(SystemExit) as caught:
                select_robot(argv, env={})
            self.assertEqual(caught.exception.code, 'Choose one robot: --h12 or --g1.')


class EnvFlagTests(unittest.TestCase):
    """Check env_flag, the sims' on/off knobs."""

    def test_unset_or_empty_is_the_default(self):
        """An unset or blank knob is its default."""
        for default in (True, False):
            self.assertIs(env_flag('K', default, env={}), default)
            self.assertIs(env_flag('K', default, env={'K': ' '}), default)

    def test_off_words_and_anything_else(self):
        """0, off, false and no in any case turn a knob off; any other value turns it on."""
        for value in ('0', 'off', 'FALSE', ' No '):
            self.assertFalse(env_flag('K', True, env={'K': value}), value)
        for value in ('1', 'on', 'yes', 'true', '2'):
            self.assertTrue(env_flag('K', False, env={'K': value}), value)


class AddRobotArgsTests(unittest.TestCase):
    """Check the --h12/--g1 options add_robot_args adds."""

    def setUp(self):
        """Make a parser with only the robot options."""
        self.parser = argparse.ArgumentParser()
        add_robot_args(self.parser)

    def test_robot_is_none_without_a_flag(self):
        """Without a flag the parsed robot is None."""
        self.assertIsNone(self.parser.parse_args([]).robot)

    def test_flags_set_robot(self):
        """Each flag sets its robot."""
        self.assertEqual(self.parser.parse_args(['--h12']).robot, 'h12')
        self.assertEqual(self.parser.parse_args(['--g1']).robot, 'g1')

    def test_flags_are_mutually_exclusive(self):
        """Both flags together are a usage error."""
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as caught:
            self.parser.parse_args(['--h12', '--g1'])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('not allowed with', err.getvalue())


class AddCommonArgsTests(unittest.TestCase):
    """Check the options add_common_args adds."""

    def setUp(self):
        """Make a parser with the common options."""
        self.parser = argparse.ArgumentParser()
        add_common_args(self.parser)

    def test_defaults(self):
        """The common options' defaults."""
        args = self.parser.parse_args([])
        self.assertEqual(vars(args), dict(robot=None, headless=False, task=None, layout=None,
                                          style=None, seed=None))

    def test_values(self):
        """Every common option parses to its value and type."""
        args = self.parser.parse_args(['--g1', '--headless', '--task', 'OpenFridge', '--layout', '9',
                                       '--style', '8', '--seed', '42'])
        self.assertEqual(vars(args), dict(robot='g1', headless=True, task='OpenFridge', layout=9,
                                          style=8, seed=42))

    def test_robot_flags_are_the_shared_pair(self):
        """The common options include the mutually exclusive robot flags."""
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser.parse_args(['--h12', '--g1'])

    def test_task_help_can_be_replaced(self):
        """A runner can replace the --task help text."""
        parser = argparse.ArgumentParser()
        add_common_args(parser, task_help='a custom task help')
        self.assertIn('a custom task help', parser.format_help())
        self.assertNotIn('a custom task help', self.parser.format_help())


class ParseArgsTests(unittest.TestCase):
    """Check parse_args' per-robot refusals."""

    def setUp(self):
        """Make a parser with the common options."""
        self.parser = argparse.ArgumentParser(prog='golem_sim.py')
        add_common_args(self.parser)

    def parse_error(self, argv, robot):
        """Return the usage error parse_args exits with: (exit code, what it wrote to stderr)."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            parse_args(self.parser, argv, robot)
        return caught.exception.code, err.getvalue()

    def test_no_flag_runs_either_robot(self):
        """Without a robot flag, either robot's runner accepts the arguments."""
        for robot in ('h12', 'g1'):
            self.assertIsNone(parse_args(self.parser, ['--headless'], robot).robot)

    def test_own_flag_is_accepted(self):
        """Each runner accepts its own robot flag."""
        self.assertEqual(parse_args(self.parser, ['--h12'], 'h12').robot, 'h12')
        self.assertEqual(parse_args(self.parser, ['--g1'], 'g1').robot, 'g1')

    def test_abbreviated_flag_that_names_the_other_robot_is_refused(self):
        """An abbreviation that argparse expands to the other robot's flag is refused."""
        # select_robot matches whole tokens, so '--g' selects the H1-2, while argparse reads '--g'
        # as '--g1'.
        self.assertEqual(select_robot(['--g'], env={}), 'h12')
        code, err = self.parse_error(['--g'], 'h12')
        self.assertEqual(code, 2)
        self.assertIn('spell the robot flag out: --g1', err)
        code, err = self.parse_error(['--h1'], 'g1')
        self.assertEqual(code, 2)
        self.assertIn('spell the robot flag out: --h12', err)

    def test_abbreviated_flag_that_names_this_robot_is_accepted(self):
        """An abbreviation of the runner's own flag is accepted."""
        self.assertEqual(parse_args(self.parser, ['--h1'], 'h12').robot, 'h12')
        self.assertEqual(parse_args(self.parser, ['--g'], 'g1').robot, 'g1')

    def test_no_timing_or_duration_option(self):
        """Neither robot's runner takes --timing or --duration."""
        for robot in ('h12', 'g1'):
            for flag in (['--timing'], ['--duration', '5']):
                code, _ = self.parse_error(flag, robot)
                self.assertEqual(code, 2)

    def test_other_options_pass_through(self):
        """Other options parse unchanged."""
        args = parse_args(self.parser, ['--task', 'OpenFridge', '--seed', '3'], 'g1')
        self.assertEqual((args.task, args.seed), ('OpenFridge', 3))


class BandLoweringFlagTests(unittest.TestCase):
    """Check golem_sim.py's --no_band_lowering: off by default, passed to sim_loop."""

    def run_main(self, argv):
        """Run golem_sim.main(argv) with the renderer check, the robot, the task and the loop stubbed; return sim_loop's kwargs."""
        from unittest import mock
        import golem_sim
        robot = mock.Mock(own_tasks=(), no_kitchen=())
        with mock.patch.object(golem_sim.sim_common, 'require_renderer'), \
                mock.patch.object(golem_sim, 'make_robot', return_value=robot), \
                mock.patch.object(golem_sim.sim_common, 'select_task', return_value='Blank'), \
                mock.patch.object(golem_sim, 'sim_loop') as loop:
            golem_sim.main(argv)
        return loop.call_args.kwargs

    def test_lowering_is_the_default(self):
        """Without the flag the sim hangs the robot and lowers it (band_lowering True)."""
        self.assertIs(self.run_main(['--h12', '--headless'])['band_lowering'], True)

    def test_flag_keeps_the_fixed_anchor(self):
        """--no_band_lowering passes band_lowering False, on either robot."""
        self.assertIs(self.run_main(['--h12', '--no_band_lowering'])['band_lowering'], False)
        self.assertIs(self.run_main(['--g1', '--headless', '--no_band_lowering'])['band_lowering'], False)
