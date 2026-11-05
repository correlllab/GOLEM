"""
Run a policy network: TorchScript or ONNX, as its spec's runtime.kind says.

Each backend imports its library when it is built, never at import time, so a robot
whose policies are all TorchScript (the Mac image has no onnxruntime) never imports
onnxruntime.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from golem_lowerbody_rl.policy.spec import RuntimeSpec

__all__ = ['OnnxRuntime', 'Runtime', 'TorchScriptRuntime', 'load_runtime']


class Runtime(Protocol):
    """Run a network on one input vector and clear its state."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the network's output (float32) for the float32 input x."""

    def reset(self) -> None:
        """Clear the network's recurrent state, if it has any."""


def _last_line(error: Exception) -> str:
    """Return the last non-empty line of error's message."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else type(error).__name__


class TorchScriptRuntime:
    """
    Run a TorchScript network on the calling thread.

    torch.set_num_threads(threads) holds for the whole process: torch's default pool made
    the H1-2 leg node take 3.6-7.5 cores and slowed the sim from 0.79x to 0.65x real
    time. The inter-op pool is set to one thread where torch still allows it (only before
    any inter-op work in the process).
    """

    def __init__(self, path: str, reset: str | None, threads: int):
        """Load the network at path; reset names its state-clearing method, if any."""
        import torch

        self._torch = torch
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # the pool exists already; its size can no longer change
        self._module = torch.jit.load(path, map_location='cpu')
        self._module.eval()
        self._path = path
        self._reset = None
        if reset is not None:
            if not hasattr(self._module, reset):
                raise ValueError(f"the network {path} has no reset method '{reset}'")
            self._reset = getattr(self._module, reset)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the network's output for x."""
        tensor = self._torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        with self._torch.no_grad():
            return self._module(tensor.unsqueeze(0)).numpy().reshape(-1)

    def reset(self) -> None:
        """Call the network's reset method, if the spec names one."""
        if self._reset is not None:
            self._reset()

    def _probe(self, size: int) -> np.ndarray:
        """Return the output for a zero input of size, with the network's buffers restored."""
        saved = [(buffer, buffer.clone()) for _, buffer in self._module.named_buffers()]
        try:
            return self(np.zeros(size, dtype=np.float32))
        finally:
            with self._torch.no_grad():
                for buffer, value in saved:
                    buffer.copy_(value)

    def check(self, input_size: int, output_size: int) -> None:
        """
        Raise ValueError unless the network takes input_size values and returns output_size.

        A TorchScript network declares no input size, so it is run once on zeros, its
        state restored after. A network that slices its input (FAME's) takes any size at
        least as long as its slices, so only its output size is checked.
        """
        try:
            out = self._probe(input_size)
        except Exception as error:  # torch.jit.Error is not a RuntimeError
            raise ValueError(f'the spec gives {input_size} inputs, which the network '
                             f'{self._path} refuses: {_last_line(error)}') from None
        if out.size != output_size:
            raise ValueError(f'the network {self._path} returns {out.size} values, '
                             f'not one per leg ({output_size})')

    def warm_up(self, input_size: int) -> None:
        """Run the network once on zeros, then restore its state."""
        self._probe(input_size)


class OnnxRuntime:
    """
    Run an ONNX network on the calling thread, CPU only.

    onnxruntime's default session pins a thread pool as wide as the machine to the same
    CPUs in every process and spins it between runs: at 50 Hz two pool threads each kept
    a CPU busy, and a second process's pool on those CPUs never finished loading. With
    threads intra-op threads (1: no pool), one inter-op thread and no spinning, nothing
    is pinned or spins; a step of HOMIE's network takes about 0.2 ms.
    """

    def __init__(self, path: str, threads: int):
        """Open an inference session on the network at path."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self._session = ort.InferenceSession(path, sess_options=options,
                                             providers=['CPUExecutionProvider'])
        self._path = path
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f'the network {path} takes {len(inputs)} inputs, not one vector')
        self._input = inputs[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the network's output for x."""
        feed = np.asarray(x, dtype=np.float32)[None, :]
        return np.asarray(self._session.run(None, {self._input.name: feed})[0][0])

    def reset(self) -> None:
        """Keep nothing: an ONNX network has no state between runs."""

    def check(self, input_size: int, output_size: int) -> None:
        """Raise ValueError unless the network declares input_size inputs, output_size outputs."""
        declared = self._input.shape[-1]
        if declared != input_size:
            raise ValueError(f'the spec gives {input_size} inputs, the network {self._path} '
                             f'takes {declared}')
        outputs = self._session.get_outputs()[0].shape[-1]
        if outputs != output_size:
            raise ValueError(f'the network {self._path} returns {outputs} values, '
                             f'not one per leg ({output_size})')

    def warm_up(self, input_size: int) -> None:
        """Run the network once on zeros: onnxruntime's first run is slow."""
        self(np.zeros(input_size, dtype=np.float32))


def load_runtime(spec: RuntimeSpec, share_dir: str, input_size: int,
                 output_size: int) -> TorchScriptRuntime | OnnxRuntime:
    """
    Return the runtime of spec's network, checked against the spec's sizes.

    The network is share_dir/spec.file. Raises ValueError for a missing file, a reset
    method the network lacks, or an input or output size the network does not have.
    With spec.warmup it is run once on zeros, then reset.
    """
    path = os.path.join(share_dir, spec.file)
    if not os.path.isfile(path):
        raise ValueError(f'the network {path} does not exist')
    runtime: TorchScriptRuntime | OnnxRuntime
    if spec.kind == 'torchscript':
        runtime = TorchScriptRuntime(path, spec.reset, spec.threads)
    else:
        runtime = OnnxRuntime(path, spec.threads)
    runtime.check(input_size, output_size)
    if spec.warmup:
        runtime.warm_up(input_size)
        runtime.reset()
    return runtime
