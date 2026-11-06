# reference/ — standalone MuJoCo deploy scripts (not built)

These are the original **standalone** policy deployment scripts, kept for
provenance and as the source of truth for the observation/action math the
policy runner implements. They are **not** part of the colcon build and are
not maintained to run as-is (they load their own local MuJoCo model and, for
the RMA script, expect the training-tree `RMA` package + an XML that is not in
this repo).

- `deploy_mujoco.py` — original walking-policy deploy → now WALK's spec,
  `config/h12/policies/walk.yaml`.
- `mujoco_deploy_h12_rma.py` — original FAME/RMA deploy → now FAME's spec,
  `config/h12/policies/fame.yaml`.

The live integration is in `golem_lowerbody_rl/`: the config-driven policy
runner `policy/` (with the trimmed inference-only encoder, `policy/encoder.py`),
which runs each policy from its spec (ALMI's, the vendored policy with no
standalone script here: `policies/almi/README.md`), and the ROS node, `node.py`.
