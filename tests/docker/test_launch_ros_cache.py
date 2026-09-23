"""Run the launcher with temporary mounts and a fake colcon (no ROS/image needed)."""
import os
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class LaunchCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = self.root / 'home/code/core_ws'
        (self.ws / 'src/demo').mkdir(parents=True)
        (self.ws / 'src/demo/package.xml').write_text('<package/>')
        (self.ws / 'install').mkdir()
        (self.ws / 'install/setup.bash').write_text(':')
        for name in ('opt/ros/humble/setup.bash', 'opt/unitree_install/lib/libunitree_sdk2.a'):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(':')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        colcon = self.bin / 'colcon'
        colcon.write_text('''#!/usr/bin/env python3
import json, os, sys
if sys.argv[1] == 'list':
    print('demo')
    if any('livox_ros_driver2' in arg for arg in sys.argv):
        print('livox_ros_driver2')
    sys.exit(int(os.environ.get('LIST_EXIT', '0')))
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(int(os.environ.get('BUILD_EXIT', '0')))
''')
        colcon.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
                        CALLS=str(self.root / 'calls'), HOME=str(self.root))
        # Relocate fixed container mount paths, keeping the executable logic intact.
        source = (REPO / 'docker/scripts/launch_ros.sh').read_text()
        self.launcher = self.root / 'launch_ros.sh'
        self.launcher.write_text(source.replace('/home/code', str(self.root / 'home/code'))
                                 .replace('/opt/', str(self.root / 'opt') + '/'))

    def run_launch(self, **env):
        return subprocess.run(['bash', str(self.launcher), 'touch', str(self.root / 'started')],
                              env=dict(self.env, **env), text=True, capture_output=True)

    def test_source_changes_trigger_incremental_build_even_with_newer_install_marker(self):
        (self.ws / 'src/demo/node.cpp').write_text('changed source')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'calls').exists(), 'colcon must scan all inputs every launch')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / 'calls').read_text().splitlines()), 2)

    def test_colcon_failure_prevents_command_execution(self):
        result = self.run_launch(BUILD_EXIT='23')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertFalse((self.root / 'started').exists())

    def test_livox_is_staged_without_running_or_mutating_upstream_build_script(self):
        driver = self.ws / 'src/livox_ros_driver2'
        driver.mkdir()
        upstream = '#!/bin/bash\nexit 0\n'
        (driver / 'build.sh').write_text(upstream)
        (driver / 'build.sh').chmod(0o755)
        (driver / 'package_ROS2.xml').write_text('<package>ROS2</package>')
        (driver / 'launch_ROS2').mkdir()
        (driver / 'launch_ROS2/example.py').write_text('launch')
        result = self.run_launch(BUILD_EXIT='19')
        self.assertEqual(result.returncode, 19, result.stderr)
        self.assertEqual((driver / 'build.sh').read_text(), upstream)
        self.assertFalse((driver / 'package.xml').exists())
        self.assertFalse((self.root / 'started').exists())
        args = json.loads((self.root / 'calls').read_text())
        self.assertIn('--symlink-install', args)
        self.assertIn('-DDISTRO_ROS=humble', args)
        paths = args[args.index('--base-paths') + 1:args.index('--cmake-args')]
        self.assertNotIn(str(driver), paths)
        staged = next(Path(p) for p in paths if Path(p).name == 'livox_ros_driver2')
        self.assertEqual((staged / 'package.xml').read_text(), '<package>ROS2</package>')

    def test_livox_refresh_removes_deleted_sources_and_preserves_other_builds(self):
        driver = self.ws / 'src/livox_ros_driver2'
        driver.mkdir()
        (driver / 'package_ROS2.xml').write_text('<package>ROS2</package>')
        (driver / 'removed.cpp').write_text('old')
        other = self.ws / 'build/demo/object.o'
        other.parent.mkdir(parents=True)
        other.write_text('keep')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.root / 'calls').read_text())
        stage = Path(args[args.index('--base-paths') + 1])
        self.assertTrue((stage / 'removed.cpp').exists())
        (driver / 'removed.cpp').unlink()
        (driver / 'added.cpp').write_text('new')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((stage / 'removed.cpp').exists())
        self.assertEqual((stage / 'added.cpp').read_text(), 'new')
        self.assertEqual(other.read_text(), 'keep')

    def test_livox_cache_moves_to_staged_source_once(self):
        driver = self.ws / 'src/livox_ros_driver2'
        driver.mkdir()
        (driver / 'package_ROS2.xml').write_text('<package>ROS2</package>')
        cache = self.ws / 'build/livox_ros_driver2/CMakeCache.txt'
        cache.parent.mkdir(parents=True)
        cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={driver}\n')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(cache.exists())
        args = json.loads((self.root / 'calls').read_text())
        stage = args[args.index('--base-paths') + 1]
        cache.parent.mkdir(parents=True)
        cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={stage}\n')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(cache.exists(), 'an already migrated cache must stay incremental')

    def test_livox_same_timestamp_source_edit_invalidates_only_driver_cache(self):
        driver = self.ws / 'src/livox_ros_driver2'
        driver.mkdir()
        (driver / 'package_ROS2.xml').write_text('<package>ROS2</package>')
        source = driver / 'node.cpp'
        source.write_text('old')
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.root / 'calls').read_text())
        stage = args[args.index('--base-paths') + 1]
        cache = self.ws / 'build/livox_ros_driver2/CMakeCache.txt'
        cache.parent.mkdir(parents=True)
        cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={stage}\n')
        unrelated = self.ws / 'build/demo/keep.o'
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text('keep')
        stamp = source.stat()
        source.write_text('new')
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(cache.exists(), 'changed driver content must invalidate older objects')
        self.assertTrue(unrelated.exists())
        self.assertEqual((Path(stage) / 'node.cpp').read_text(), 'new')

    def test_removed_package_install_prevents_launch(self):
        marker = self.ws / 'install/removed/share/colcon-core/packages/removed'
        marker.parent.mkdir(parents=True)
        marker.write_text('')
        result = self.run_launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('removed', result.stderr)
        self.assertFalse((self.root / 'started').exists())

    def test_package_discovery_failure_prevents_launch(self):
        result = self.run_launch(LIST_EXIT='17')
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertFalse((self.root / 'started').exists())

    def test_existing_backdated_mjpc_cache_is_revalidated_once(self):
        src = self.root / 'home/code/mujoco_mpc'
        build = src / 'build'
        build.mkdir(parents=True)
        source = src / 'CMakeLists.txt'
        source.write_text('project(test)')
        os.utime(source, (946684800, 946684800))
        (build / 'CMakeCache.txt').write_text('cache')
        obj = build / 'cached.o'
        obj.write_text('object')
        rebuild = self.root / 'home/code/h12_sim_scripts/rebuild_mjpc.sh'
        rebuild.parent.mkdir()
        rebuild.write_text('#!/bin/bash\nexit 0\n')
        rebuild.chmod(0o755)
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(source.stat().st_mtime_ns, obj.stat().st_mtime_ns)
        stamp = source.stat().st_mtime_ns
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(source.stat().st_mtime_ns, stamp)

    @unittest.skipUnless(all(shutil.which(x) for x in ('colcon', 'cmake', 'c++')),
                         'requires local colcon, cmake, and C++ compiler')
    def test_real_build_recompiles_cpp_and_generated_input_and_rejects_removed_package(self):
        # Use a tiny standalone CMake package, exercising actual dependency
        # checks and installed executables without ROS or the Livox SDK.
        (self.bin / 'colcon').unlink()
        shutil.rmtree(self.ws / 'src/demo')
        driver = self.ws / 'src/livox_ros_driver2'
        driver.mkdir()
        (driver / 'package_ROS2.xml').write_text(
            '<package format="3"><name>livox_ros_driver2</name><version>0.0.1</version>'
            '<description>fixture</description><maintainer email="t@example.com">Test</maintainer>'
            '<license>MIT</license><export><build_type>cmake</build_type></export></package>')
        (driver / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.10)\nproject(livox_ros_driver2)\n'
            'configure_file(value.msg value.h COPYONLY)\n'
            'add_executable(fixture node.cpp)\n'
            'target_include_directories(fixture PRIVATE ${CMAKE_CURRENT_BINARY_DIR})\n'
            'install(TARGETS fixture DESTINATION bin)\n')
        source = driver / 'node.cpp'
        source.write_text('#include <iostream>\n#include "value.h"\nint main(){std::cout << VALUE;}\n')
        message = driver / 'value.msg'
        message.write_text('#define VALUE 1\n')
        binary = self.ws / 'install/livox_ros_driver2/bin/fixture'
        for value in (1, 2):
            message.write_text(f'#define VALUE {value}\n')
            result = self.run_launch()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(subprocess.check_output([str(binary)], text=True), str(value))
        stamp = source.stat()
        source.write_text(source.read_text().replace('<< VALUE', '<< VALUE + 1'))
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(subprocess.check_output([str(binary)], text=True), '3')
        shutil.rmtree(driver)
        (self.root / 'started').unlink()
        result = self.run_launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale installed package', result.stderr)
        self.assertFalse((self.root / 'started').exists())

    def test_dirty_mjpc_source_is_newer_than_hydrated_objects(self):
        src = self.root / 'home/code/mujoco_mpc'
        seed = self.root / 'opt/mjpc-build-seed'
        src.mkdir(parents=True)
        seed.mkdir(parents=True)
        (src / 'CMakeLists.txt').write_text('project(test)')
        dirty = src / 'dirty.cpp'
        dirty.write_text('original')
        for args in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Test', '-c', 'user.email=t@example.com', 'commit', '-qm', 'seed']):
            subprocess.run(['git', '-C', str(src), *args], check=True, capture_output=True)
        ref = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True)
        (seed / '.mjpc_ref').write_text(ref)
        (seed / 'CMakeCache.txt').write_text('cache')
        obj = seed / 'dirty.o'
        obj.write_text('old binary')
        dirty.write_text('dirty edits')
        os.utime(dirty, (946684800, 946684800))
        rebuild = self.root / 'home/code/h12_sim_scripts/rebuild_mjpc.sh'
        rebuild.parent.mkdir()
        rebuild.write_text('#!/bin/bash\nexit 0\n')
        rebuild.chmod(0o755)
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(dirty.stat().st_mtime_ns, (src / 'build/dirty.o').stat().st_mtime_ns)


if __name__ == '__main__':
    unittest.main()
