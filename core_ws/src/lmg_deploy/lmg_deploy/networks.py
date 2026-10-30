"""The exported networks: two actors and the pelvis-motion estimator, as TorchScript.

scripts/skrl/export_marl.py in Locomanipulation_game writes them. Each actor
file already contains its observation scaler and the +-10 action clamp, so it
maps a raw observation straight to the raw action the env would apply; the
estimator file contains its input normalization.

An LSTM actor (export_marl.py's RecurrentActor: forward(obs, h, c) ->
(action, h, c), with hidden_size and num_layers attributes) is wrapped by
RecurrentTorchScriptNetwork, which carries the hidden state from step to step;
reset() zeroes it, as training zeroed it at every episode start.
load_network picks the wrapper.

ZeroNetwork is a stand-in for the unit tests: legs hold the default pose, arms
follow pure IK, and the estimator reports no pelvis motion. The node always
loads the TorchScript exports.
"""

import numpy as np
import torch


class TorchScriptNetwork:
    def __init__(self, path: str, num_inputs: int, num_outputs: int):
        self.module = torch.jit.load(path, map_location="cpu")
        self.module.eval()
        with torch.inference_mode():
            out = self.module(torch.zeros(1, num_inputs))
        if out.shape != (1, num_outputs):
            raise ValueError(f"{path}: {num_inputs} inputs gave output shape {tuple(out.shape)}, "
                             f"expected (1, {num_outputs})")

    def reset(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = self.module(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
        return out.squeeze(0).numpy().astype(np.float64)


class RecurrentTorchScriptNetwork:
    def __init__(self, path: str, num_inputs: int, num_outputs: int):
        self.module = torch.jit.load(path, map_location="cpu")
        self.module.eval()
        self.state_shape = (int(self.module.num_layers), 1, int(self.module.hidden_size))
        self.reset()
        with torch.inference_mode():
            out, _, _ = self.module(torch.zeros(1, num_inputs), self._h, self._c)
        if out.shape != (1, num_outputs):
            raise ValueError(f"{path}: {num_inputs} inputs gave output shape {tuple(out.shape)}, "
                             f"expected (1, {num_outputs})")

    def reset(self) -> None:
        self._h = torch.zeros(self.state_shape)
        self._c = torch.zeros(self.state_shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out, self._h, self._c = self.module(torch.from_numpy(x.astype(np.float32)).unsqueeze(0), self._h, self._c)
        return out.squeeze(0).numpy().astype(np.float64)


def load_network(path: str, num_inputs: int, num_outputs: int):
    """TorchScriptNetwork, or RecurrentTorchScriptNetwork for an exported LSTM actor."""
    if hasattr(torch.jit.load(path, map_location="cpu"), "hidden_size"):
        return RecurrentTorchScriptNetwork(path, num_inputs, num_outputs)
    return TorchScriptNetwork(path, num_inputs, num_outputs)


class ZeroNetwork:
    def __init__(self, num_outputs: int):
        self._out = np.zeros(num_outputs)

    def reset(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._out.copy()
