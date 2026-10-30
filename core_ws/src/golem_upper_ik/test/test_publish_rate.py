import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / 'tools' / 'publish_rate_probe.py'


def test_lowcmd_publish_holds_500hz():
    env = dict(os.environ, ROS_DOMAIN_ID='92')
    done = subprocess.run(
        [sys.executable, str(PROBE), '--rate', '500', '--seconds', '10'],
        env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    stats = json.loads(done.stdout.strip().splitlines()[-1])
    print(json.dumps(stats))
    assert stats['mean_hz'] >= 495, stats
    assert stats['p99_period_ms'] <= 4.0, stats
    assert stats['max_gap_ms'] <= 20.0, stats
