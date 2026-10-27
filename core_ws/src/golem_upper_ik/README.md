# golem_upper_ik

The upper-body (waist and arms) IK controller shared by the Unitree H1-2 and
G1. It solves Pink IK at 30 Hz and publishes `unitree_hg/msg/LowCmd` to
`golem_safety` at 500 Hz, from one Python process. `tools/publish_rate_probe.py`
measures whether `rclpy` holds that publish rate beside an IK-sized CPU load.
