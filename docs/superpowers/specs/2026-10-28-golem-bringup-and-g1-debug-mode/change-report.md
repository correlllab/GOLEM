# Change report: golem_bringup, golem_robocasa and the G1's debug-mode start-up

The changes a user of either robot must know, one row per change, with the task that makes it.

| Task | Robot | Before | After | Why / what to do |
|---|---|---|---|---|
| 1 | H1-2, G1 | The RoboCasa sim lived in `h1_robocasa/` and was bind-mounted at `/home/code/h1_robocasa`. | It lives in `golem_robocasa/` and is mounted at `/home/code/golem_robocasa`. `h12_mujoco.py` and every module inside keep their names, and the sim does the same as before. | The directory is the sim of both robots, not the H1-2's. Restart the `golem_sim_robocasa` container after pulling: a running one keeps the old mount, and a stack that sets `PYTHONPATH=h1_robocasa` must name `golem_robocasa`. An image built before the rename still runs, with its `WORKDIR` unchanged, since the launchers `cd` to the new path; a rebuild is not needed. |
