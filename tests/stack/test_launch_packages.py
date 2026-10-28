"""The stack test's launch files start golem_safety and golem_upper_ik.

None names h12_safety_layer, h12_ros2_controller, g1_safety_layer or g1_control.
File parsing only, so it runs anywhere:
    python3 -m unittest discover -s tests/stack -p 'test_*.py' -v
"""
import ast
from pathlib import Path
import re
import unittest

STACK = Path(__file__).resolve().parent
OLD_PACKAGES = ('h12_safety_layer', 'h12_ros2_controller', 'g1_safety_layer', 'g1_control')
OLD_NAME = re.compile(r'\b(?:' + '|'.join(OLD_PACKAGES) + r')\b')


def node_packages(tree):
    """The package= values of the Node(...) calls in a parsed launch file."""
    return {keyword.value.value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'Node'
            for keyword in node.keywords
            if keyword.arg == 'package' and isinstance(keyword.value, ast.Constant)}


class LaunchPackagesTests(unittest.TestCase):
    def test_no_launch_names_old_packages(self):
        files = sorted(STACK.glob('*.launch.py'))
        self.assertTrue(files, f'no *.launch.py in {STACK}')
        named = {}
        for path in files:
            source = path.read_text(encoding='utf-8')
            ast.parse(source, filename=str(path))
            old = sorted(set(OLD_NAME.findall(source)))
            if old:
                named[path.name] = old
        self.assertEqual(named, {})

    def test_minimal_stack_starts_golem_safety_and_golem_upper_ik(self):
        tree = ast.parse((STACK / 'minimal_stack.launch.py').read_text(encoding='utf-8'))
        self.assertLessEqual({'golem_safety', 'golem_upper_ik'}, node_packages(tree))


if __name__ == '__main__':
    unittest.main()
