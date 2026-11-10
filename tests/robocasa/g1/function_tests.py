"""Collect a G1 test file's test_* functions into one unittest.TestCase, so that unittest runs them.

The G1 files write their tests as plain functions with assert statements. Each ends with

    G1FooTests = function_tests.test_case(globals(), "G1FooTests")

    if __name__ == "__main__":
        unittest.main()

so that `python3 test_g1_foo.py` (run_in_container.sh) and unittest's loaders run every test_*
function as a test of that class. The class belongs to the file's module, so the file's
setUpModule and tearDownModule run around it.
"""
import unittest


def test_case(namespace, name):
    """Return a unittest.TestCase subclass called name whose test methods call namespace's test_* functions.

    namespace is the test file's globals(); each method keeps its function's name and docstring.
    """
    def method(fn):
        """Return a test method that calls fn."""
        def run(self):
            fn()
        run.__name__, run.__doc__ = fn.__name__, fn.__doc__
        return run

    tests = {key: method(fn) for key, fn in namespace.items() if key.startswith("test_") and callable(fn)}
    cls = type(name, (unittest.TestCase,), tests)
    cls.__module__ = namespace["__name__"]
    cls.__qualname__ = name
    return cls
