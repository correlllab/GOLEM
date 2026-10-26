'''CLI entrypoint for the G1 safety layer'''

import time
import argparse
import threading

import sys
from pathlib import Path

# put project root and the sibling g1_description package in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT.parent / 'g1_description'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from g1_safety_layer.core import console
from g1_safety_layer.core.exit_signals import ExitRequest
from g1_safety_layer.core.safety_layer import SafetyLayer
from g1_safety_layer.core.config import init_channel_factory_guard, load_config

def _ensure_yaml_extension(config_path: Path) -> Path:
    '''Append .yaml when config path has no suffix'''
    if config_path.suffix:
        return config_path
    return config_path.with_suffix('.yaml')

def _resolve_config_path(config_name: str) -> Path:
    '''Resolve config from either bare name or config/<name> path'''
    config_path = _ensure_yaml_extension(Path(config_name))
    if config_path.is_absolute():
        return config_path
    if config_path.parts and config_path.parts[0] == 'config':
        return PROJECT_ROOT / config_path
    return PROJECT_ROOT / 'config' / config_path

def main(config_name: str) -> int:
    '''Start the low_cmd safety relay until an exit signal; return the exit code'''
    config_path = _resolve_config_path(config_name)
    config = load_config(config_path)
    init_channel_factory_guard(config)
    safety_layer = SafetyLayer(config)
    # SIGINT, SIGTERM and SIGHUP hand control back before the process ends
    exit_request = ExitRequest()
    exit_request.install()
    failed = []

    def run() -> None:
        try:
            safety_layer.start()
        except Exception as exc:
            failed.append(exc)
            console.write_line(f'Safety layer exited with error: {exc}')

    worker = threading.Thread(target=run, name='safety_layer_worker', daemon=True)
    worker.start()
    while not exit_request.is_set() and worker.is_alive():
        time.sleep(0.1)
    if exit_request.is_set():
        console.write_line(f'{exit_request.signal_name} received, handing control back...')
        reason = f'Exit requested ({exit_request.signal_name})'
    else:
        reason = 'Safety layer stopped'
    handed_back = safety_layer.shutdown(reason)
    worker.join(timeout=1.0)
    # without an exit signal the safety layer stopped by itself: a failure, as
    # is a hand-back that did not finish in time
    clean = exit_request.is_set() and not failed and handed_back
    return console.finish(0 if clean else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='G1 low_cmd safety relay')
    parser.add_argument(
        '--config',
        type=str,
        default='arm_sdk.yaml',
        help='Config name or path, e.g. arm_sdk.yaml or config/split.yaml',
    )
    args = parser.parse_args()
    sys.exit(main(args.config))
