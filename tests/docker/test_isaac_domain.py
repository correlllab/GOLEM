"""Exercise Isaac's real DDS initialization without requiring Isaac/CycloneDDS."""
import ast
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class IsaacDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim/src/python/dds/common/dds_master.py'
        tree = ast.parse(path.read_text())
        manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DDSManager')
        method = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == '_init_dds')
        cls.code = compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec')

    def initialize(self, value):
        calls = []
        namespace = {'os': os, 'initialize_transport': calls.append}
        exec(self.code, namespace)
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': value}), contextlib.redirect_stdout(io.StringIO()):
            result = namespace['_init_dds'](SimpleNamespace(dds_initialized=False))
        return result, calls

    def test_configured_domain_reaches_sdk(self):
        self.assertEqual(self.initialize('7'), (True, [7]))

    def test_empty_domain_defaults_to_simulation(self):
        self.assertEqual(self.initialize(''), (True, [1]))

    def test_invalid_domains_abort_initialization(self):
        for domain in ('0', '00', '-1', '233', 'bad'):
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                self.initialize(domain)


if __name__ == '__main__':
    unittest.main()
