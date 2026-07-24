"""ROS 2 node for mapping joystick inputs to UAV commands and flight modes.

Flight Modes:
    - AUTO: Standard "offline" simulation.
    - MANUAL:
        1. Stores the last automatic state.
        2. Maintains hover if there is no input.
        3. Maps controller output to commands.
        4. Command -> Control -> Dynamics flow.
"""

from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import TwistStamped
from m300_msgs.msg import FlightMode
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy


class JoyMapper(Node):
    """Maps joystick inputs to velocity commands and flight mode toggles.

    Subscribes to joystick and telemetry topics to generate manual flight
    commands and manage transitions between AUTO and MANUAL modes.

    Attributes:
        current_state (np.ndarray): UAV current state [x, y, z, yaw].
        positive_yaw (int): Button state for positive yaw command.
        negative_yaw (int): Button state for negative yaw command.
        flight_mode (int): Current active flight mode.
    """

    def __init__(self) -> None:
        """Initializes the JoyMapper node, publishers, and subscribers."""
        super().__init__("joy_mapper")

        self.joy_sub = self.create_subscription(
            Joy, "/joy", self.joy_callback, 10
        )
        self.fmd_pub = self.create_publisher(
            FlightMode, "/m300_sim/flight_mode", 10
        )
        self.cmd_pub = self.create_publisher(
            TwistStamped, "/m300_sim/manual_cmd", 10
        )
        self.dynamic_sub = self.create_subscription(
            Odometry,
            "/m300_sim/telemetry_topic",
            self.dynamic_callback,
            10,
        )

        self.current_state = np.zeros(4)
        self.positive_yaw = 0
        self.negative_yaw = 0
        self.flight_mode = FlightMode.AUTO

    def dynamic_callback(self, msg: Odometry) -> None:
        """Updates the current state based on incoming telemetry.

        Args:
            msg (Odometry): The incoming odometry message containing pose.
        """
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
        """Applies a deadzone filter to a joystick axis value.

        Args:
            value (float): The raw axis value from the joystick.
            deadzone (float): The threshold below which the value is zeroed.

        Returns:
            float: The scaled and filtered axis value.
        """
        if abs(value) < deadzone:
            return 0.0

        sign = 1.0 if value > 0 else -1.0
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def joy_callback(self, msg: Joy) -> None:
        """Processes joystick inputs and publishes flight commands.

        Reads axis and button states, applies deadzones, maps them to physical
        velocities, and checks for flight mode toggles.

        Args:
            msg (Joy): The incoming joystick message.
        """
        flight_mode_ax = msg.axes[7]

        if flight_mode_ax == 1.0:
            self.flight_mode = FlightMode.MANUAL
        elif flight_mode_ax == -1.0:
            self.flight_mode = FlightMode.AUTO

        dz = 0.1

        ax_z_vel = -self.apply_deadzone(msg.axes[4], dz)
        ax_north_vel = self.apply_deadzone(msg.axes[1], dz)
        
        # Negative sign flips axis direction (-1 -> West; 1 -> East)
        ax_east_vel = -self.apply_deadzone(msg.axes[0], dz)

        self.positive_yaw = msg.buttons[5]
        self.negative_yaw = msg.buttons[4]
        ax_yaw_orientation = float(self.positive_yaw - self.negative_yaw)

        # Build and publish velocity command
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = "base_link"

        cmd_msg.twist.linear.x = float(ax_north_vel)
        cmd_msg.twist.linear.y = float(ax_east_vel)
        cmd_msg.twist.linear.z = float(ax_z_vel)
        cmd_msg.twist.angular.z = ax_yaw_orientation

        self.cmd_pub.publish(cmd_msg)

        # Build and publish flight mode
        fmd_msg = FlightMode()
        fmd_msg.mode = self.flight_mode
        self.fmd_pub.publish(fmd_msg)


def main(args: Optional[List[str]] = None) -> None:
    """Initializes and spins the JoyMapper ROS 2 node.

    Args:
        args (Optional[List[str]]): Command line arguments.
    """
    rclpy.init(args=args)
    node = JoyMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()