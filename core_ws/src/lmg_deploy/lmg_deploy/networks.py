"""Load the exported actor networks (TorchScript) and run them on numpy observations.

scripts/skrl/export_marl.py in Locomanipulation_game writes them. Each actor
file contains its observation scaler and the +-10 action clamp, so it maps a
raw observation straight to the raw action the env would apply.

An LSTM actor (export_marl.py's RecurrentActor: forward(obs, h, c) ->
(action, h, c), with hidden_size and num_layers attributes) is wrapped by
RecurrentTorchScriptNetwork, which carries the hidden state from step to step;
reset() zeroes it, as training zeroed it at every episode start.
load_network picks the wrapper.
"""

import numpy as np
import torch


class TorchScriptNetwork:
    """Run a feed-forward TorchScript actor on the CPU."""

    def __init__(self, path: str, num_inputs: int, num_outputs: int):
        """Load the actor at `path`; raise ValueError unless num_inputs zeros give num_outputs values."""
        self.module = torch.jit.load(path, map_location="cpu")
        self.module.eval()
        with torch.inference_mode():
            out = self.module(torch.zeros(1, num_inputs))
        if out.shape != (1, num_outputs):
            raise ValueError(f"{path}: {num_inputs} inputs gave output shape {tuple(out.shape)}, "
                             f"expected (1, {num_outputs})")

    def reset(self) -> None:
        """Do nothing: a feed-forward actor has no state."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the raw action for one observation vector."""
        with torch.inference_mode():
            out = self.module(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
        return out.squeeze(0).numpy().astype(np.float64)


class RecurrentTorchScriptNetwork:
    """Run an LSTM TorchScript actor on the CPU, carrying its hidden state between calls."""

    def __init__(self, path: str, num_inputs: int, num_outputs: int):
        """Load the actor at `path`; raise ValueError unless num_inputs zeros give num_outputs values."""
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
        """Zero the hidden and cell state, as at an episode start."""
        self._h = torch.zeros(self.state_shape)
        self._c = torch.zeros(self.state_shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the raw action for one observation vector and advance the hidden state."""
        with torch.inference_mode():
            out, self._h, self._c = self.module(torch.from_numpy(x.astype(np.float32)).unsqueeze(0), self._h, self._c)
        return out.squeeze(0).numpy().astype(np.float64)


def load_network(path: str, num_inputs: int, num_outputs: int):
    """Return a TorchScriptNetwork, or a RecurrentTorchScriptNetwork for an exported LSTM actor."""
    if hasattr(torch.jit.load(path, map_location="cpu"), "hidden_size"):
        return RecurrentTorchScriptNetwork(path, num_inputs, num_outputs)
    return TorchScriptNetwork(path, num_inputs, num_outputs)
