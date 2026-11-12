"""ROS 2 joystick teleoperation node for Xbox controller input.

Maps gamepad thumbsticks, shoulder bumpers, D-pad, and face buttons into UAV
velocity commands, flight mode transitions (AUTO / MANUAL), and real-time
neural network residual compensation toggles.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import TwistStamped
from m300_msgs.msg import FlightMode
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class JoystickNode(Node):
    """Translate Xbox controller ``Joy`` messages into UAV velocity and mode commands.

    Attributes:
        current_state (np.ndarray): Latest ``[x, y, z, yaw]`` state from odometry.
        flight_mode (int): Active flight mode (``FlightMode.AUTO`` or ``FlightMode.MANUAL``).
        nn_enabled (bool): Current state of neural network residual compensation.
        deadzone (float): Thumbstick deadzone threshold.
    """

    def __init__(self) -> None:
        super().__init__('joystick_node')

        self.fmd_pub = self.create_publisher(
            FlightMode, '/m300_sim/flight_mode', 10
        )
        self.cmd_pub = self.create_publisher(
            TwistStamped, '/m300_sim/cmd', 10
        )
        self.nn_pub = self.create_publisher(
            Bool, '/m300_sim/nn_enable', 10
        )

        self.joy_sub = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10
        )
        self.dynamic_sub = self.create_subscription(
            Odometry,
            '/m300_sim/telemetry_topic',
            self.dynamic_callback,
            10,
        )
        self.nn_sub = self.create_subscription(
            Bool,
            '/m300_sim/nn_enable',
            self.nn_sync_callback,
            10,
        )

        self.current_state = np.zeros(4, dtype=np.float64)
        self.positive_yaw = 0
        self.negative_yaw = 0
        self.flight_mode = FlightMode.AUTO
        self.nn_enabled = False
        self._prev_btn_a = 0
        self.deadzone = 0.1

    def nn_sync_callback(self, msg: Bool) -> None:
        """Synchronize local neural network state when toggled via the GUI."""
        self.nn_enabled = bool(msg.data)

    def dynamic_callback(self, msg: Odometry) -> None:
        """Update position and yaw heading from incoming UAV odometry."""
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y
        z_pos = msg.pose.pose.position.z
        self.current_state[0:3] = [x_pos, y_pos, z_pos]

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        rot = Rotation.from_quat([qx, qy, qz, qw])
        _, _, yaw = rot.as_euler('xyz')
        self.current_state[3] = yaw

    def apply_deadzone(
        self,
        value: float,
        deadzone: float = 0.2,
        expo: float = 0.35,
    ) -> float:
        """Apply deadzone and linear-cubic exponential shaping to a thumbstick axis.

        Args:
            value: Raw normalized axis reading in ``[-1.0, 1.0]``.
            deadzone: Minimum absolute input required to produce an output.
            expo: Cubic blending factor in ``[0.0, 1.0]`` for fine center control.

        Returns:
            Shaped axis command in ``[-1.0, 1.0]``.
        """
        if abs(value) < deadzone:
            return 0.0

        sign = 1.0 if value > 0.0 else -1.0
        norm_val = (abs(value) - deadzone) / (1.0 - deadzone)
        return sign * ((1.0 - expo) * norm_val + expo * (norm_val**3))

    def _set_nn_enabled(self, enabled: bool) -> None:
        """Publish a neural network toggle message when the state changes."""
        if enabled != self.nn_enabled:
            self.nn_enabled = enabled
            msg = Bool()
            msg.data = self.nn_enabled
            self.nn_pub.publish(msg)
            state_label = 'ENABLED' if self.nn_enabled else 'DISABLED'
            self.get_logger().info(
                f'[Xbox] Neural Network compensation: {state_label}'
            )

    def joy_callback(self, msg: Joy) -> None:
        """Process Xbox controller axes and buttons at each ``/joy`` update."""
        if len(msg.axes) > 7:
            flight_mode_ax = msg.axes[7]
            if flight_mode_ax == 1.0:
                self.flight_mode = FlightMode.MANUAL
            elif flight_mode_ax == -1.0:
                self.flight_mode = FlightMode.AUTO

        # Toggle Neural Network with Button A (rising edge) or D-Pad Left/Right
        if len(msg.buttons) > 0:
            btn_a = int(msg.buttons[0])
            if btn_a == 1 and self._prev_btn_a == 0:
                self._set_nn_enabled(not self.nn_enabled)
            self._prev_btn_a = btn_a

        if len(msg.axes) > 6:
            dpad_lr = msg.axes[6]
            if dpad_lr < -0.5:
                self._set_nn_enabled(True)
            elif dpad_lr > 0.5:
                self._set_nn_enabled(False)

        ax_z_vel = -self.apply_deadzone(msg.axes[4], self.deadzone)
        ax_north_vel = self.apply_deadzone(msg.axes[1], self.deadzone)
        ax_east_vel = -self.apply_deadzone(msg.axes[0], self.deadzone)

        self.positive_yaw = msg.buttons[5]
        self.negative_yaw = msg.buttons[4]
        ax_yaw_orientation = float(self.positive_yaw - self.negative_yaw)

        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'

        cmd_msg.twist.linear.x = float(ax_north_vel)
        cmd_msg.twist.linear.y = float(ax_east_vel)
        cmd_msg.twist.linear.z = float(ax_z_vel)
        cmd_msg.twist.angular.z = ax_yaw_orientation

        self.cmd_pub.publish(cmd_msg)

        fmd_msg = FlightMode()
        fmd_msg.mode = self.flight_mode
        self.fmd_pub.publish(fmd_msg)


def main(args: list[str] | None = None) -> None:
    """Initialize and spin the Xbox controller teleoperation ROS 2 node."""
    rclpy.init(args=args)
    node = JoystickNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
