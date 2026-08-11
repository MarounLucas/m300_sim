from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import TwistStamped
from m300_msgs.msg import FlightMode
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy


class JoystickNode(Node):
    def __init__(self) -> None:
        super().__init__("joystick_node")

        # ===== PUBLISHERS ===== #
        self.fmd_pub = self.create_publisher(
            FlightMode, "/m300_sim/flight_mode", 10
        )
        self.cmd_pub = self.create_publisher(
            TwistStamped, "/m300_sim/cmd", 10
        )

        # ===== SUBSCRIBERS ===== #
        self.joy_sub = self.create_subscription(
            Joy, "/joy", self.joy_callback, 10
        )
        self.dynamic_sub = self.create_subscription(
            Odometry,
            "/m300_sim/telemetry_topic",
            self.dynamic_callback,
            10
        )

        # Init parameters
        self.current_state = np.zeros(4)
        self.positive_yaw = 0
        self.negative_yaw = 0
        self.flight_mode = FlightMode.AUTO # auto fmd as standard
        self.deadzone = 0.1

    def dynamic_callback(self, msg: Odometry) -> None:
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y
        z_pos = msg.pose.pose.position.z
        self.current_state[0:3] = [x_pos, y_pos, z_pos]

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        rot = Rotation.from_quat([qx, qy, qz, qw])
        _, _, yaw = rot.as_euler("xyz")
        self.current_state[3] = yaw

    def apply_deadzone(self, value: float, deadzone: float = 0.1) -> float:
        if abs(value) < deadzone:
            return 0.0

        sign = 1.0 if value > 0 else -1.0
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def joy_callback(self, msg: Joy) -> None:
        # Flight mode  
        flight_mode_ax = msg.axes[7] # Up/Down 

        if flight_mode_ax == 1.0:
            self.flight_mode = FlightMode.MANUAL
        elif flight_mode_ax == -1.0:
            self.flight_mode = FlightMode.AUTO

        # ===== Velocity Axes ===== #
        # Z: Right joystick up (-) / down (+)
        ax_z_vel = -self.apply_deadzone(msg.axes[4], self.deadzone)

        # North: Left joystick up (-) / down (+)
        ax_north_vel = self.apply_deadzone(msg.axes[1], self.deadzone)
        
        # East: Left joystick left (-) / right (+)
        ax_east_vel = -self.apply_deadzone(msg.axes[0], self.deadzone)

        # Yaw orientation
        self.positive_yaw = msg.buttons[5] # RB
        self.negative_yaw = msg.buttons[4] # LB
        ax_yaw_orientation = float(self.positive_yaw - self.negative_yaw)


        # ===== PUB MSG ===== #
        # building cmd msg
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = "base_link"

        cmd_msg.twist.linear.x = float(ax_north_vel)
        cmd_msg.twist.linear.y = float(ax_east_vel)
        cmd_msg.twist.linear.z = float(ax_z_vel)
        cmd_msg.twist.angular.z = ax_yaw_orientation

        self.cmd_pub.publish(cmd_msg)

        # building fmd msg
        fmd_msg = FlightMode()
        fmd_msg.mode = self.flight_mode
        self.fmd_pub.publish(fmd_msg)


def main(args = None) -> None:
    rclpy.init(args=args)
    node = JoystickNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()