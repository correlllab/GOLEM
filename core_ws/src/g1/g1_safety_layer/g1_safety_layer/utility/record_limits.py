'''Record low_state ranges for each joint'''

import time
import argparse
import numpy as np

import sys
from pathlib import Path

# put project root in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_safety_layer.core.joint_limits import (
    JOINT_NAMES,
    MOTOR_COUNT,
    URDF_POSITION_LIMITS,
    URDF_VELOCITY_LIMITS,
    URDF_TORQUE_LIMITS,
)


METRIC_KEYS = ('q', 'dq', 'ddq', 'tau')
DATA_DIR = PROJECT_ROOT / 'data' / 'low_state_record'

def _extract_metrics(msg: LowState_) -> dict[str, np.ndarray]:
    '''Extract q dq ddq tau arrays from low_state message'''
    return {
        'q': np.asarray([float(msg.motor_state[i].q)
                         for i in range(MOTOR_COUNT)], dtype=np.float32),
        'dq': np.asarray([float(msg.motor_state[i].dq)
                          for i in range(MOTOR_COUNT)], dtype=np.float32),
        'ddq': np.asarray([float(msg.motor_state[i].ddq)
                           for i in range(MOTOR_COUNT)], dtype=np.float32),
        'tau': np.asarray([float(msg.motor_state[i].tau_est)
                           for i in range(MOTOR_COUNT)], dtype=np.float32),
    }

def _print_limits(mins: dict[str, np.ndarray], maxs: dict[str, np.ndarray]) -> None:
    '''Print per-joint min and max for each metric'''
    for key in METRIC_KEYS:
        print(f'[{key}]')
        for i, joint_name in enumerate(JOINT_NAMES):
            print(f'  {i:02d} {joint_name}: min={mins[key][i]: .6f} max={maxs[key][i]: .6f}')
        print('')


def _print_limits_format(mins: dict[str, np.ndarray], maxs: dict[str, np.ndarray]) -> None:
    '''Print yaml-ready limits derived from recorded bounds and urdf limits'''
    q_lower_offset = []
    q_upper_offset = []
    q_offset = []
    dq_ratio = []
    tau_ratio = []

    for i in range(MOTOR_COUNT):
        q_low = float(URDF_POSITION_LIMITS[i]['low'])
        q_high = float(URDF_POSITION_LIMITS[i]['high'])
        q_lower_offset.append(float(mins['q'][i]) - q_low)
        q_upper_offset.append(q_high - float(maxs['q'][i]))
        q_offset.append(min(q_lower_offset[-1], q_upper_offset[-1]))

        dq_peak = max(abs(float(mins['dq'][i])), abs(float(maxs['dq'][i])))
        dq_ratio.append(dq_peak / float(URDF_VELOCITY_LIMITS[i]))

        tau_peak = max(abs(float(mins['tau'][i])), abs(float(maxs['tau'][i])))
        tau_ratio.append(tau_peak / float(URDF_TORQUE_LIMITS[i]))

    # print('q_offset_from_lower:')
    # for i, joint_name in enumerate(JOINT_NAMES):
    #     print(f'  - {q_lower_offset[i]:.6f} # {joint_name}')

    # print('q_offset_from_upper:')
    # for i, joint_name in enumerate(JOINT_NAMES):
    #     print(f'  - {q_upper_offset[i]:.6f} # {joint_name}')

    print('q_offset:')
    for i, joint_name in enumerate(JOINT_NAMES):
        print(f'  - {q_offset[i]:.6f} # {joint_name}')

    print('dq_ratio:')
    for i, joint_name in enumerate(JOINT_NAMES):
        print(f'  - {dq_ratio[i]:.6f} # {joint_name}')

    print('tau_ratio:')
    for i, joint_name in enumerate(JOINT_NAMES):
        print(f'  - {tau_ratio[i]:.6f} # {joint_name}')

def _print_limits_from_load_name(load_name: str) -> None:
    '''Load npz by name and print limits'''
    data_path = DATA_DIR / f'{load_name}.npz'
    if not data_path.exists():
        raise FileNotFoundError(f'File not found: {data_path}')

    with np.load(data_path) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float32) for k in METRIC_KEYS}

    # assert correct shapes
    for key in METRIC_KEYS:
        if arrays[key].ndim != 2 or arrays[key].shape[1] != MOTOR_COUNT:
            raise ValueError(
                f'Invalid shape for {key}: {arrays[key].shape}, expected (N, {MOTOR_COUNT})'
            )

    count = int(arrays['q'].shape[0])
    if count == 0:
        print(f'Loaded 0 samples from {data_path}')
        return

    # print mins & maxs
    mins = {k: np.min(v, axis=0) for k, v in arrays.items()}
    maxs = {k: np.max(v, axis=0) for k, v in arrays.items()}
    _print_limits(mins, maxs)
    _print_limits_format(mins, maxs)

def _record_low_state(interface: str, domain_id: int, topic: str, save_name: str | None) -> None:
    '''Subscribe to low_state and print limits on exit'''
    stats = {'sample_count': 0}
    mins = {k: np.full((MOTOR_COUNT,), np.inf, dtype=np.float32) for k in METRIC_KEYS}
    maxs = {k: np.full((MOTOR_COUNT,), -np.inf, dtype=np.float32) for k in METRIC_KEYS}
    buffers = {k: [] for k in METRIC_KEYS}

    def _on_message(msg: LowState_) -> None:
        stats['sample_count'] += 1
        values = _extract_metrics(msg)
        # record min, max and all the data
        for key in METRIC_KEYS:
            mins[key] = np.minimum(mins[key], values[key])
            maxs[key] = np.maximum(maxs[key], values[key])
            if save_name:
                buffers[key].append(values[key])

    if interface:
        ChannelFactoryInitialize(domain_id, interface)
    else:
        ChannelFactoryInitialize(domain_id)

    subscriber = ChannelSubscriber(topic, LowState_)
    subscriber.Init(_on_message, 10)

    print(f'Recording low_state from topic={topic}')
    print('Press ctrl+c to stop and print joint limits')

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.Close()

    sample_count = stats['sample_count']
    print(f'Received {sample_count} samples')
    if sample_count == 0:
        return

    # print recorded limits to terminal
    _print_limits(mins, maxs)
    _print_limits_format(mins, maxs)
    # save as npz if needed
    if save_name:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data_path = DATA_DIR / f'{save_name}.npz'
        np.savez(
            data_path,
            q=np.stack(buffers['q'], axis=0),
            dq=np.stack(buffers['dq'], axis=0),
            tau=np.stack(buffers['tau'], axis=0),
            ddq=np.stack(buffers['ddq'], axis=0),
        )
        print(f'Saved {sample_count} samples to {data_path}')


def main(interface: str,
         domain_id: int,
         topic: str,
         save_name: str | None,
         load_name: str | None) -> None:
    '''Run recorder in live mode or load mode'''
    if load_name:
        _print_limits_from_load_name(load_name)
    else:
        _record_low_state(interface=interface,
                          domain_id=domain_id,
                          topic=topic,
                          save_name=save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='record low_state and print per-joint limits')
    parser.add_argument('--interface', type=str, default='', help='DDS network interface')
    parser.add_argument('--domain-id', type=int, default=0, help='DDS domain id')
    parser.add_argument('--topic', type=str, default='rt/lowstate', help='low_state topic name')
    parser.add_argument('--save', type=str, default=None, help='save as data/low_state_record/<name>.npz')
    parser.add_argument('--load', type=str, default=None, help='load data/low_state_record/<name>.npz and print limits')
    args = parser.parse_args()

    if args.save and args.load:
        raise ValueError('use --save or --load, not both')

    main(
        interface=args.interface,
        domain_id=args.domain_id,
        topic=args.topic,
        save_name=args.save,
        load_name=args.load,
    )
