"""Check that the wrappers' container lookups cannot match an image of the same name."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ContainerCheckTests(unittest.TestCase):
    """Check the `docker inspect` calls in the wrapper scripts."""

    def test_inspect_by_container_name_is_typed(self):
        """Every `docker inspect "$NAME"` in a wrapper passes `--type container`."""
        # golem_ros, golem_sim_robocasa and golem_sim_isaac each name both a container and an
        # image, and an untyped `docker inspect` succeeds on the image alone.
        for path in sorted((ROOT / 'docker/scripts').glob('*.sh')):
            for line in path.read_text().splitlines():
                if re.search(r'docker inspect\b[^|;]*"\$(?:NAME|name)"', line):
                    self.assertIn('--type container', line, f'{path.name}: {line.strip()}')


if __name__ == '__main__':
    unittest.main()
