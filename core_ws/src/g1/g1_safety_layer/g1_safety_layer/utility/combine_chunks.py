'''Combine chunked npz logs into one merged npz'''

import argparse
import numpy as np
from pathlib import Path


def _list_chunks(base_dir: Path, prefix: str) -> list[Path]:
    return sorted(base_dir.glob(f'{prefix}*.npz'))

def combine_chunks(base_dir: str, prefix: str, output: str) -> Path:
    '''Combine all matching chunk files into one npz'''
    base = Path(base_dir)
    files = _list_chunks(base, prefix)
    if not files:
        raise FileNotFoundError(f'no chunk files found in {base} with prefix {prefix}')

    merged: dict[str, list[np.ndarray]] = {}
    key_order: list[str] | None = None

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            keys = list(data.keys())
            # initialize list for merged file
            if key_order is None:
                key_order = keys
                for key in key_order:
                    merged[key] = []
            # check that keys match previous files
            elif keys != key_order:
                raise ValueError(f'key mismatch in {file_path.name}')

            for key in key_order:
                merged[key].append(np.asarray(data[key]))

    out_data: dict[str, np.ndarray] = {}
    for key, chunks in merged.items():
        out_data[key] = np.concatenate(chunks, axis=0)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_data)
    return out_path

def main(load: str, prefix: str, save: str) -> None:
    '''Run chunk combiner cli'''
    out = combine_chunks(load, prefix, save)
    print(f'combined chunks into {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine chunked npz logs')
    parser.add_argument('--load', type=str, default='/tmp/g1_safety_layer')
    parser.add_argument('--prefix', type=str, default='chunk_')
    parser.add_argument('--save', type=str, default='/tmp/g1_safety_layer/combined.npz')
    args = parser.parse_args()
    main(args.load, args.prefix, args.save)
