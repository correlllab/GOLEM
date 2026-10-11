"""The exported networks: two actors and the pelvis-motion estimator, as TorchScript.

scripts/skrl/export_marl.py in Locomanipulation_game writes them. Each actor
file already contains its observation scaler and the +-10 action clamp, so it
maps a raw observation straight to the raw action the env would apply; the
estimator file contains its input normalization.

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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = self.module(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
        return out.squeeze(0).numpy().astype(np.float64)


class ZeroNetwork:
    def __init__(self, num_outputs: int):
        self._out = np.zeros(num_outputs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._out.copy()
