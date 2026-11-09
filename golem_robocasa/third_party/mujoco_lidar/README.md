# MuJoCo-LiDAR's MID-360 scan pattern

`mid360.npy` is the Livox MID-360's non-repetitive scan pattern from
[MuJoCo-LiDAR](https://github.com/TATP-233/MuJoCo-LiDAR), file
`src/mujoco_lidar/scan_mode/mid360.npy` at commit
`e6b879df4b96daaed30ef10fe02650d29048fa9b` (`main` on 2026-11-08), copied byte
for byte (sha256 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`).
`LICENSE` is that repository's licence at the same commit (MIT, Copyright (c)
2025 Yufei Jia), which covers the file.

The file is a NumPy array, float32, 800,000 rows x 2 columns: azimuth (rad,
0 to 2 pi) and elevation (rad, -0.126 to 0.910, positive up in the sensor's own
frame), one row per point in firing order. 20,000 consecutive rows are one
10 Hz frame at the sensor's 200,000 points/s, so the file holds 40 frames
before it repeats. `golem_robocasa/mid360.py` reads it (`load_pattern`).
