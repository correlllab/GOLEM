# Policies: one per robot

Each robot has one folder, and it holds exactly one policy: the iteration in use.

| Folder | What it holds |
|---|---|
| `h12/` | the H1-2's policy: `h12.yaml` (the deploy config), `export.yaml` (what the export writes; its `source` names the run and checkpoint), `legs.pt`, `arms.pt` |
| `g1/` | the G1's policy, the same files with `g1.yaml`; empty until there is one |

**Keep only one iteration at a time; delete old ones.** A new export replaces the robot's
folder: delete the old files and export into it (`export_marl.py --out .../policies/h12`, see
`../README.md` "Exporting a checkpoint"). Do not add folders beside it for older or candidate
iterations. Candidates live in Locomanipulation_game's `logs/`, and an older iteration comes
back from git history or by re-exporting the checkpoint its `export.yaml` names.

The launches take the robot's folder; there is no `policy:=` to choose between iterations. The
`.pt` files are in Git LFS (`.gitattributes`).
