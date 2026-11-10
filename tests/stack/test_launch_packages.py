"""Check that the stack test's launch file starts golem_safety and golem_upper_ik.

It only parses the file, so it runs anywhere:
    python3 -m unittest discover -s tests/stack -p 'test_*.py' -v
"""
import ast
from pathlib import Path
import unittest

STACK = Path(__file__).resolve().parent


def node_packages(tree):
    """Return the `package=` values of the Node(...) calls in a parsed launch file."""
    return {keyword.value.value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'Node'
            for keyword in node.keywords
            if keyword.arg == 'package' and isinstance(keyword.value, ast.Constant)}


class LaunchPackagesTests(unittest.TestCase):
    """Check the packages minimal_stack.launch.py starts."""

    def test_minimal_stack_starts_golem_safety_and_golem_upper_ik(self):
        """minimal_stack.launch.py starts nodes from golem_safety and golem_upper_ik."""
        tree = ast.parse((STACK / 'minimal_stack.launch.py').read_text(encoding='utf-8'))
        self.assertLessEqual({'golem_safety', 'golem_upper_ik'}, node_packages(tree))


if __name__ == '__main__':
    unittest.main()
