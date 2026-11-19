"""Check the Unitree SDK's DDS and a ROS node can share one DDS domain in one process.

Usage: python3 check_unitree_dds.py [--ros]
  Without --ros: open a Unitree rt/lowstate publisher (the SDK's own CycloneDDS).
  --ros (ROS sourced): import rclpy first, as the sims do, then open the Unitree
  publisher and create a ROS node with a publisher on the same domain, as the sims'
  sensor bridge does.
Run by docker/mac/BaseDockerfile, so an image whose two CycloneDDS copies collide
(a segfault in the first Unitree topic, or a ROS domain the SDK already holds)
fails its build. Exit status: 0 when every step works; otherwise the failing
step's error or signal.
"""
import faulthandler
import os
import sys

# A domain no sim or real robot uses, so a build host's DDS traffic is not joined.
DOMAIN = 232


def main() -> None:
    """Run the steps; raise on the first that fails."""
    faulthandler.enable()
    ros = "--ros" in sys.argv[1:]
    if ros:
        os.environ["ROS_DOMAIN_ID"] = str(DOMAIN)
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Empty
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(DOMAIN)
    ChannelPublisher("rt/lowstate", LowState_).Init()
    if ros:
        rclpy.init()
        node = Node("check_unitree_dds")
        node.create_publisher(Empty, "check_unitree_dds", 1)
        node.destroy_node()
        rclpy.shutdown()
    print(f"check_unitree_dds: ok{' (with a ROS node)' if ros else ''}")


if __name__ == "__main__":
    main()
