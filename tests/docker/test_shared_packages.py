"""Check that the packages both robots run name no robot in their code.

golem_safety, golem_upper_ik, golem_slam and golem_lowerbody_rl serve both robots: what
differs per robot is data (golem_safety's `robots/<robot>.yaml` profiles, each package's
`config/<robot>/`), never a branch or a literal in code. The robot names are the profiles golem_safety ships.
"""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'core_ws/src'

# The code only: not the data (config/, robots/), the tests or the tools.
SCANNED = [
    SRC / 'golem_safety/src',
    SRC / 'golem_safety/include',
    SRC / 'golem_safety/launch',
    SRC / 'golem_upper_ik/golem_upper_ik',
    SRC / 'golem_upper_ik/launch',
    SRC / 'golem_upper_ik/setup.py',
    SRC / 'golem_slam/golem_slam',
    SRC / 'golem_slam/launch',
    SRC / 'golem_slam/setup.py',
    SRC / 'golem_lowerbody_rl/golem_lowerbody_rl',
    SRC / 'golem_lowerbody_rl/launch',
    SRC / 'golem_lowerbody_rl/setup.py',
]
CODE_SUFFIXES = {'.py', '.cpp', '.hpp', '.h', '.cc'}


def robots():
    """Return the robot names, one per golem_safety robot profile."""
    return sorted(path.stem for path in (SRC / 'golem_safety/robots').glob('*.yaml'))


def _names_a_robot(text, names):
    """Return whether `text` is a robot name or starts with one followed by `_` or `/`."""
    return any(text == name or text.startswith((name + '_', name + '/')) for name in names)


def _compared_name(node):
    """Return the identifier an expression ends in (`robot` for `self.robot`), else ''."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ''


def _literal_strings(node):
    """Return the string literals of a constant or a tuple, list or set of constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [s for element in node.elts for s in _literal_strings(element)]
    return []


def python_hits(source, names):
    """Return (line, what) for each robot name in Python code: a string literal or a comparison."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _names_a_robot(node.value, names):
                hits.append((node.lineno, f'string literal {node.value!r}'))
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any('robot' in _compared_name(operand) for operand in operands) and any(
                    _literal_strings(operand) for operand in operands):
                hits.append((node.lineno, 'comparison of a robot against a literal'))
    return hits


# A C++ comment, string or character literal, whichever starts first.
_CPP_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)
_CPP_COMPARISON = re.compile(
    r'\brobot\w*\s*[!=]=\s*"|"\s*[!=]=\s*[\w.>-]*robot\b|\brobot\w*\s+in\s*[({]', re.I)


def _line(source, offset):
    """Return the 1-based line number of a character offset in `source`."""
    return source.count('\n', 0, offset) + 1


def cpp_hits(source, names):
    """Return (line, what) for each robot name in C++ code: a string literal or a comparison."""
    # Comments are blanked to spaces so that the line numbers stay those of `source`.
    code = _CPP_TOKEN.sub(
        lambda m: re.sub(r'[^\n]', ' ', m.group()) if m.group()[0] == '/' else m.group(), source)
    hits = [(_line(code, m.start()), f'string literal {m.group()}')
            for m in _CPP_TOKEN.finditer(code)
            if m.group()[0] == '"' and _names_a_robot(m.group()[1:-1], names)]
    hits += [(_line(code, m.start()), 'comparison of a robot against a literal')
             for m in _CPP_COMPARISON.finditer(code)]
    return sorted(hits)


def name_hits(relative, names):
    """Return a message for each part of a path (file, module or directory) named for a robot."""
    return [f'{part!r} is named for a robot' for part in Path(relative).parts
            if _names_a_robot(part.split('.')[0], names)]


def scanned_files():
    """Yield every code file under `SCANNED`."""
    for entry in SCANNED:
        if entry.is_file():
            yield entry
        else:
            for path in sorted(entry.rglob('*')):
                if path.is_file() and path.suffix in CODE_SUFFIXES and '__pycache__' not in path.parts:
                    yield path


def violations(names=None):
    """Return every robot name found in the scanned paths' names and code."""
    names = robots() if names is None else names
    found = []
    for path in scanned_files():
        relative = path.relative_to(ROOT)
        package = SRC / path.relative_to(SRC).parts[0]
        found += [f'{relative}: {what}' for what in name_hits(path.relative_to(package), names)]
        finder = python_hits if path.suffix == '.py' else cpp_hits
        found += [f'{relative}:{line}: {what}' for line, what in finder(path.read_text(), names)]
    return found


class SharedPackagesCarryNoRobotNamesTests(unittest.TestCase):
    """Scan the shared packages for robot names."""

    def test_the_robots_are_the_profiles(self):
        """golem_safety ships a profile for each of the H1-2 and the G1."""
        self.assertTrue({'h12', 'g1'} <= set(robots()), robots())

    def test_every_scanned_path_exists(self):
        """Every scanned path exists and the scan covers more than 20 files."""
        for entry in SCANNED:
            self.assertTrue(entry.exists(), entry)
        self.assertGreater(len(list(scanned_files())), 20)

    def test_no_robot_name_in_code(self):
        """No shared code names a robot in a path, a string literal or a comparison."""
        found = violations()
        self.assertEqual(found, [], 'robot names in shared code:\n' + '\n'.join(found))


class ScannerTests(unittest.TestCase):
    """Check the scanner on small sources."""

    NAMES = ['h12', 'g1']

    def test_python_string_literals(self):
        """A Python literal equal to or prefixed by a robot name is found; one merely containing it is not."""
        source = 'a = "h12"\nb = "g1_legs"\nc = f"h12/{x}"\nd = "other"\ne = "h12 or g1"\n'
        self.assertEqual([line for line, _ in python_hits(source, self.NAMES)], [1, 2, 3])

    def test_python_comparisons(self):
        """A comparison of a `robot` name against literals is found; other comparisons are not."""
        source = ('if robot == "x": pass\n'
                  'if "x" != self.robot: pass\n'
                  'if robot in ("x", "y"): pass\n'
                  'if robot == other: pass\n'
                  'if n == "x": pass\n')
        self.assertEqual([line for line, _ in python_hits(source, self.NAMES)], [1, 2, 3])

    def test_python_comments_are_not_scanned(self):
        """Robot names in Python comments are ignored."""
        self.assertEqual(python_hits('# h12 and "g1"\nx = 1\n', self.NAMES), [])

    def test_cpp_string_literals_and_comparisons(self):
        """C++ literals and comparisons are found and names in comments are ignored."""
        source = ('// "h12" in a comment\n'
                  'auto a = "g1";\n'
                  'auto b = "h12_x";\n'
                  'if (robot == "x") {}\n'
                  'if ("x" != profile.robot) {}\n'
                  '/* "g1"\n */ auto c = "fine";\n')
        self.assertEqual([line for line, _ in cpp_hits(source, self.NAMES)], [2, 3, 4, 5])

    def test_names(self):
        """A file named for a robot is found; a robot-neutral name is not."""
        self.assertEqual(name_hits('golem_slam/h12_real_slam.launch.py', self.NAMES),
                         ["'h12_real_slam.launch.py' is named for a robot"])
        self.assertEqual(name_hits('golem_slam/golem_slam/g1_body_to_pelvis.py', self.NAMES),
                         ["'g1_body_to_pelvis.py' is named for a robot"])
        self.assertEqual(name_hits('golem_slam/launch/navigation.launch.py', self.NAMES), [])


if __name__ == '__main__':
    unittest.main()
