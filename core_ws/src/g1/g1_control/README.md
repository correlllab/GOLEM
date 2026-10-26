# g1_control

Upper-body IK controller for the Unitree G1 (29 DoF, Dex3-1 hands) with ROS 2
action servers.

Ported from [correlllab/h12_ros2_controller](https://github.com/correlllab/h12_ros2_controller)
(H1-2), whose history is kept in this repo.
Joint names, slot order and URDF limits come from `g1_description`.

## What it does

- Solves Pink IK on the 14 arm joints; the end effectors are the Dex3-1 palms
    (`left_hand_palm_link`, `right_hand_palm_link`).
- Holds the 3 waist joints at `controller.waist_target` (default 0): every
    command carries the hold angles, never the measured waist the IK was seeded
    from, so the hold does not drift with the waist's sag. Legs are never
    commanded.
- Publishes only to `g1_safety_layer` (`rt/safety/lowcmd_upper_in`), which alone
    writes `rt/arm_sdk` or `rt/lowcmd`. A config that targets any other topic is
    rejected.
- Its own estop and shutdown stop publishing rather than sending zero-gain
    commands: the safety layer sees the upper channel go stale after 2 s and, in
    `arm_sdk` mode, hands the waist and arms back to Unitree's controller.
- On start it moves the arms to `home` (`utility/named_config.py`), then the
    waist to its hold angles (see Start-up).
- The G1 has no self-collision model yet: IK runs without a self-collision
    barrier, and OMPL planning (`plan=true` goals) aborts with a clear message.

## Start-up

- It enables nothing until a fresh `rt/lowstate` with finite joint positions
    arrives. After `controller.state_timeout` (5 s) it stops with an error that
    names the DDS domain and interface.
- Every slot's first command is the measured pose, with gravity compensation.
    Nothing starts from the all-zero pose.
- A follow gate (`core/follow_gate.py`) then holds each waist and arm command
    within `controller.follow_tolerance` (0.05 rad, well below the safety
    layer's 0.15 rad engage gate) of the measured joint, so a command never
    runs ahead of a robot that does not execute it: in split mode before the
    walking node commands, or in `arm_sdk` mode while the weight ramps in. The
    IK, the waist hold and the steady-state target restart from the gated
    command at every step, so nothing jumps when the gate lets go.
- The gate opens when the controller sees the safety layer relay its
    commands. It reads, and never writes, the safety layer's output
    (`topics.safety_out`): the gate opens on the first sample that carries the
    commands on every waist and arm slot, within `follow_tolerance` and with
    kp > 0, at weight 1 on `rt/arm_sdk` (`g1.yaml`, `sim.yaml`; the end of the
    1 s ramp) or on `rt/lowcmd` (`*_split.yaml`). The safety layer's estop
    command (kp 0) never opens it, and neither does how the measured joints
    move.
- Once open, the gate never moves a command again: a held joint resists a
    push or a tipping torso instead of following it. Planned paths
    (`execute_path`) are refused until then.
- The start-up re-sends the gated hold, near the measured pose, until the gate
    opens; only then does it move the arms to `home` and the waist to its hold
    angles. Each step has its own budget: the gate must open within
    `controller.startup_timeout` (30 s), and the moves must be done within
    `controller.startup_move_timeout` (30 s) of the gate opening, so a late
    engagement does not cut the move short. Otherwise the controller stops with
    an error; the safety layer then hands back (`arm_sdk`) or estops (split).
- `follow_tolerance: 0` turns the gate off.

## Named postures

All values are in `utility/named_config.py` and were checked with forward
kinematics on the G1 model.

- `home`: all zeros; upper arms down, elbows at 90 deg, forearms forward.
- `arms_down`: straight arms hanging below the shoulders.
- `t_pose`: straight arms out to the sides (the target of Unitree's `rt/arm_sdk` example).
- `arms_front_45`: straight arms 45 deg forward and down.

## Configs

- `config/g1.yaml`: real robot, DDS domain 0, with the safety layer's
    `arm_sdk.yaml`.
- `config/g1_split.yaml`: real robot with the safety layer's `split.yaml`
    (HOMIE's walking policy on the legs): `g1.yaml` with a stiff waist, and
    the follow gate reads `rt/lowcmd`.
- `config/sim.yaml`, `config/sim_split.yaml`: the same in simulation (with
    `sim_arm_sdk.yaml`, `sim_split.yaml`), DDS domain 1, relaxed estop limits
    and faster arm speeds.
- The real configs' speeds are deliberately conservative for the first
    sessions: `dq_lim` 0.3 rad/s, `v_lim` 0.15 m/s, `w_lim` 0.5 rad/s (the sim
    keeps 1.0 rad/s, 0.5 m/s, 1.0 rad/s), and a 10 s default goal timeout.
- Gains are per joint. The arms keep Unitree's `rt/arm_sdk` example values
    (kp 60, kd 1.5). The waist kd gives a damping ratio kd / (2 sqrt(kp I)) of
    about 0.4-0.5 with the URDF's waist inertia I at home (yaw 0.39, roll 0.71,
    pitch 0.72 kg m^2): kd 4.5, 6, 6 at kp 60, and 8, 10, 10 at the split
    configs' kp 200 (kd 10 is the safety layer's `kd_max`). The integral term is
    off. Gains and estop limits are starting values, not tuned on our G1 yet.
- `controller.imu_frame` (default `imu_in_pelvis`) names the frame of the
    `rt/lowstate` IMU; set `imu_in_torso` if turning the waist changes the IMU yaw
    while the pelvis stays still.

## ROS Interface

- Build the workspace with `g1_description`, `g1_safety_layer` and
    `custom_ros_messages`, then `source install/setup.bash`.
- Start the safety layer and the controller together. The launch file checks
    the pairing with `utility/config_pairing.py`, as `g1_bringup` does: the
    safety config must relay the upper channel, and each one runs with one
    controller config (`arm_sdk` with `g1`, `split` with `g1_split`,
    `sim_arm_sdk` with `sim`, `sim_split` with `sim_split`):

    ```bash
    ros2 launch g1_control robot_safety_launch.py config:=g1 safety_config:=arm_sdk
    ```

- Or run the servers beside a safety layer started on its own. Nothing reaches
    the robot without it, and a server that does not see it relay its commands
    within `controller.startup_timeout` stops with an error:

    ```bash
    ros2 run g1_control frame_task_server --config g1.yaml
    ros2 run g1_control dual_arm_server --config g1.yaml
    ```

- Interactive clients send named postures or frame targets; press BACKSPACE to
    cancel the active goal:

    ```bash
    ros2 run g1_control frame_task_client
    ros2 run g1_control dual_arm_client
    ```

- From the command line:

    ```bash
    ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
      "{config_name: 't_pose', plan: false, duration: {sec: 0, nanosec: 0}}"
    ```

## Launch Files

- [robot_safety_launch.py](launch/robot_safety_launch.py) starts `robot_state_publisher`, `joint_state_publisher`, `g1_safety_layer/safety_node` and `frame_task_server`.
- [robot_launch.py](launch/robot_launch.py) starts `robot_state_publisher`, `joint_state_publisher` and `frame_task_server`.
- [full_launch.py](launch/full_launch.py) starts `robot_state_publisher`, `joint_state_publisher`, `frame_task_server` and `rviz2`.
- [robot_tf_launch.py](launch/robot_tf_launch.py) starts `robot_state_publisher` and `joint_state_publisher`.
- [desktop_launch.py](launch/desktop_launch.py) starts `rviz2` with the default RViz config.
- `joint_state_publisher` publishes the 29 body joints; the Dex3-1 finger joints
    will come from the hand driver.

## Files

- `config/`: YAML configuration files.
- `g1_control/`: source code.
    - `core/`: robot model, IK solver, DDS channel interface, low-cmd handler,
        follow gate, planner.
    - `core/controller/`: upper, arm, frame and impedance controllers.
    - `ros2/`: action servers, clients and the joint state publisher.
    - `utility/`: joint definitions, limits, named postures, config loading and
        pairing, paths.
    - `example/`: scripts run without ROS (`uv run` or `python3` from this
        directory). The ones that publish commands (the arm and frame
        controller examples, `motor_viewer*.py --damping`) refuse DDS domain 0
        unless given `--real-robot`, and say why.
- `launch/`, `rviz/`: launch files and the RViz config.
- `test/`: unit tests, plus closed-loop DDS tests of the controller against
    kinematic fake G1s, alone and through the real safety layer in `arm_sdk`
    and split mode.

## Remaining hazards

Not handled in code; keep them in mind on the real robot.

- The follow gate opens for good once the safety layer relays the commands.
    After any safety layer estop (latched until restart) or restart, restart
    the controller too: a fresh safety layer would meet a command held
    wherever the robot used to be.
- Until the gate opens, a joint pushed or sagging more than 0.05 rad from its
    command drags the command along. In `arm_sdk` mode that lasts through the
    1 s weight ramp, while Unitree's share of the blend still holds the joint.
- The gate takes a relayed command for an executed one: it opens when the
    safety layer writes the controller's commands (`rt/arm_sdk` at weight 1,
    or `rt/lowcmd` in split mode), and does not check that the measured joints
    follow them. If the robot does not apply them at that moment (for example,
    Unitree's controller is not yet in a mode that honors `rt/arm_sdk`), the
    move to `home` runs open loop and the arms jump once the commands take
    effect. Start the controller only when the robot already applies the
    safety layer's output (`rt/arm_sdk` or `rt/lowcmd`).
- The gate trusts the safety layer's output topic: a controller config whose
    `topics.safety_out` is not the safety layer's `topics.low_cmd_out` never
    sees it engage, and the start-up stops after `startup_timeout`. Another
    writer on that topic whose sample carries the controller's commands would
    open it.
- The arm IK models the waist frozen at 0 (the reduced model): a non-zero
    `waist_target`, or the waist's sag, offsets the palms from their targets.
- No self-collision model: the IK can drive the arms into the torso or each
    other.
- Gravity compensation is recomputed only while the controller steps (goals
    and start-up); between goals the last torque is held. The torso then tips
    the waist roll and pitch with about 23 N*m/rad (arms at home; 34 with the
    arms out), so at rest their hold is softer than kp: about 37 N*m/rad of 60.
- The waist's damping ratio changes with the arms' pose: about 0.46 at home,
    0.26 on yaw and roll with the arms out to the sides (`t_pose`).
- The controller's own limit checker compares `rt/lowstate` with its estop
    limits but does not reject non-finite values; the safety layer does, once
    engaged.
- Config pairing is by name only: a renamed copy of a sim config passes as a
    real one, and a custom controller config is not matched to the safety
    config.
- With `--real-robot`, `motor_viewer*.py --damping` sends kp 0 on every slot:
    in `arm_sdk` mode the waist goes limp under the upper body while Unitree
    balances the legs.
- A controller built with `init=False` skips the start-up: it does not wait
    for the gate, and the waist stays held where it was measured instead of
    at `waist_target`. Every server and example uses `init=True`.
- `joint_state_publisher` publishes zeros until `rt/lowstate` arrives (RViz
    only; it commands nothing).

## Tests

```bash
PYTHONPATH=.:../g1_description python3 -m pytest test/test_g1_controller.py \
    test/test_controller_config.py test/test_reduced_joint_planner.py test/test_controller_loop_dds.py
```

The DDS tests run on domain 97 inside one process; they never touch the robot.
They need the workspace built and sourced (`g1_safety_layer` for the arm_sdk
test).
`uv` users: `pyproject.toml` and `uv.lock` are inherited from the H1-2 package and
have not been re-locked for G1.
