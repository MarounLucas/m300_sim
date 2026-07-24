"""ROS 2 multi-threaded node for the UAV cascade controller.

This module orchestrates the execution of the cascade controller across 
different frequencies (50Hz, 250Hz, and 1000Hz) using a MultiThreadedExecutor 
and Mutual Exclusion Locks to prevent race conditions during state updates.
"""

import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import TwistStamped, WrenchStamped
from m300_msgs.msg import FlightMode
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

from .control import CascadeController


@dataclass
class AircraftModel:
    """Temporary data structure to fulfill CascadeController initialization."""
    mass: float = 0.0
    max_ascent_speed: float = 0.0
    max_descent_speed: float = 0.0
    cruise_speed: float = 0.0
    max_tilt_angle: float = 0.0
    max_roll_pitch_rate: float = 0.0
    max_yaw_rate: float = 0.0


class Controller(Node):
    """Orchestrates the UAV's multi-frequency cascade control loops.

    Manages ROS 2 subscriptions (telemetry, trajectory, manual commands) and 
    publishes the calculated control efforts (wrenches) at 1000Hz using 
    thread-safe callback groups.

    Attributes:
        controller (CascadeController): The underlying mathematical controller.
        current_fmd (int): The active flight mode (AUTO or MANUAL).
        state (np.ndarray): The 12-element current physical state vector.
    """

    def __init__(self) -> None:
        """Initializes the controller node, parameters, and threaded timers."""
        super().__init__("controller_node")

        self.current_fmd = FlightMode.AUTO
        self.manual_cmd = [0.0, 0.0, 0.0, 0.0]

        # Declare necessary control limits
        self.declare_parameters(
            namespace='',
            parameters=[
                ('mass', 6.47),
                ('max_ascent_speed', -6.0),
                ('max_descent_speed', 5.0),
                ('cruise_speed', 9.2),
                ('max_tilt_angle', 0.523),
                ('max_roll_pitch_rate', 5.23),
                ('max_yaw_rate', 1.74)
            ]
        )

        ac_model = AircraftModel(
            mass=float(self.get_parameter('mass').value),
            max_ascent_speed=float(self.get_parameter('max_ascent_speed').value),
            max_descent_speed=float(self.get_parameter('max_descent_speed').value),
            cruise_speed=float(self.get_parameter('cruise_speed').value),
            max_tilt_angle=float(self.get_parameter('max_tilt_angle').value),
            max_roll_pitch_rate=float(self.get_parameter('max_roll_pitch_rate').value),
            max_yaw_rate=float(self.get_parameter('max_yaw_rate').value)
        )

        # Instantiate the controller with YAML data
        self.controller = CascadeController(ac_model)

        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.state = np.zeros(12)
        
        self.total_thrust = 0.0
        self.target_torque = np.zeros(3)

        self.lock = threading.Lock()

        # Thread-safe Callback groups
        self.cb_groups_subs = MutuallyExclusiveCallbackGroup()
        self.cb_group_50hz = MutuallyExclusiveCallbackGroup()
        self.cb_group_250hz = MutuallyExclusiveCallbackGroup()
        self.cb_group_1000hz = MutuallyExclusiveCallbackGroup()

        # ==== Subscribers ==== #
        self.traj_sub = self.create_subscription(
            Odometry, '/m300_sim/trajectory_topic',
            self.trajectory_callback, 10, callback_group=self.cb_groups_subs
        )
        
        self.dyn_sub = self.create_subscription(
            Odometry, '/m300_sim/telemetry_topic',
            self.state_callback, 10, callback_group=self.cb_groups_subs
        )

        self.fmd_sub = self.create_subscription(
            FlightMode, "/m300_sim/flight_mode", self.fmd_callback, 10
        )

        self.cmd_sub = self.create_subscription(
            TwistStamped, "/m300_sim/manual_cmd", self.cmd_callback, 10
        )

        self.start_sub = self.create_subscription(
            Empty, '/m300_sim/start_mission', self.start_callback, 10
        )

        # ==== Publisher ==== #
        self.ctrl_pub = self.create_publisher(
            WrenchStamped, '/m300_sim/control_topic', 10
        )

        # ==== Timers ==== #
        self.create_timer(1 / 50.0, self.loop_50hz, callback_group=self.cb_group_50hz)
        self.create_timer(1 / 250.0, self.loop_250hz, callback_group=self.cb_group_250hz)
        self.create_timer(1 / 1000.0, self.loop_1000hz, callback_group=self.cb_group_1000hz)

    def start_callback(self, msg: Empty) -> None:
        """Handles the mission start signal by synchronizing limits.

        Args:
            msg (Empty): The trigger message.
        """
        self.get_logger().info("Synchronizing updated control parameters from GUI...")
        self.reload_parameters()

    def reload_parameters(self) -> None:
        """Loads and applies physical control limits from the YAML file dynamically."""
        pkg_share = get_package_share_directory('m300_sim')
        ac_yaml = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')

        try:
            with open(ac_yaml, 'r', encoding='utf-8') as file:
                ac_data = yaml.safe_load(file)['controller_node']['ros__parameters']

                with self.lock:
                    self.controller.mass = ac_data['mass']
                    self.controller.min_z_vel = ac_data['max_ascent_speed']
                    self.controller.max_z_vel = ac_data['max_descent_speed']
                    self.controller.cruise_vel = ac_data['cruise_speed']
                    self.controller.max_tilt = ac_data['max_tilt_angle']
                    self.controller.max_roll_pitch_rate = ac_data['max_roll_pitch_rate']
                    self.controller.max_yaw_rate = ac_data['max_yaw_rate']

                    self.controller.max_rate = np.array([
                        ac_data['max_roll_pitch_rate'],
                        ac_data['max_roll_pitch_rate'],
                        ac_data['max_yaw_rate']
                    ])
        except Exception as err:
            self.get_logger().error(f"Failed to load new control limits: {err}")

    def loop_50hz(self) -> None:
        """Executes the slow outer loops (Position and Velocity) at 50Hz."""
        with self.lock:
            if self.current_fmd == FlightMode.AUTO:
                self.controller._xy_pos_control()
                self.controller._z_pos_control()

            elif self.current_fmd == FlightMode.MANUAL:
                x_cmd = self.manual_cmd[0]
                y_cmd = self.manual_cmd[1]
                z_cmd = self.manual_cmd[2]
                yaw_cmd = self.manual_cmd[3]

                self.controller.des_velocity[0] = x_cmd * 2.5
                self.controller.des_velocity[1] = y_cmd * 2.5
                self.controller.des_velocity[2] = z_cmd * 2.5

                yaw_rate = float(self.get_parameter('max_yaw_rate').value)
                dt = 1.0 / 50.0

                self.controller.des_angles[2] += (yaw_cmd * yaw_rate) * dt

            self.controller._xy_vel_control(dt=1.0 / 50.0)
            self.controller._z_vel_control(dt=1.0 / 50.0)

            # Note: _accel_to_attitude returns the required thrust (u_total)
            self.total_thrust = self.controller._accel_to_attitude()

    def loop_250hz(self) -> None:
        """Executes the intermediate attitude control loop at 250Hz."""
        with self.lock:
            self.controller._angle_control()

    def loop_1000hz(self) -> None:
        """Executes the fast inner angular rate loop and publishes efforts at 1000Hz."""
        with self.lock:
            self.target_torque = self.controller._angular_rate_control(dt=1.0 / 1000.0)

            ctrl_msg = WrenchStamped()
            ctrl_msg.header.stamp = self.get_clock().now().to_msg()
            ctrl_msg.header.frame_id = 'base_link'

            ctrl_msg.wrench.force.z = float(self.total_thrust)
            ctrl_msg.wrench.torque.x = float(self.target_torque[0])
            ctrl_msg.wrench.torque.y = float(self.target_torque[1])
            ctrl_msg.wrench.torque.z = float(self.target_torque[2])

            self.ctrl_pub.publish(ctrl_msg)

    def state_callback(self, msg: Odometry) -> None:
        """Parses telemetry to update the controller's current knowledge of the drone.

        Args:
            msg (Odometry): The current state of the UAV.
        """
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.state[9:12] = px, py, pz

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        rot = Rotation.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler('xyz')
        self.state[6:9] = roll, pitch, yaw

        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z
        self.state[0:3] = u, v, w

        p = msg.twist.twist.angular.x
        q = msg.twist.twist.angular.y
        r = msg.twist.twist.angular.z
        self.state[3:6] = p, q, r

        with self.lock:
            self.controller.update_state(self.state)

    def trajectory_callback(self, msg: Odometry) -> None:
        """Parses the desired trajectory to update the controller's targets.

        Args:
            msg (Odometry): The desired state provided by the trajectory planner.
        """
        self.target_pos[0] = msg.pose.pose.position.x
        self.target_pos[1] = msg.pose.pose.position.y
        self.target_pos[2] = msg.pose.pose.position.z

        quat_x = msg.pose.pose.orientation.x
        quat_y = msg.pose.pose.orientation.y
        quat_z = msg.pose.pose.orientation.z
        quat_w = msg.pose.pose.orientation.w

        rot = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w])
        _, _, self.target_yaw = rot.as_euler('xyz')

        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_z = msg.twist.twist.linear.z

        vel = np.array([vel_x, vel_y, vel_z])
        waypoint = np.array([
            self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_yaw
        ])

        with self.lock:
            if self.current_fmd == FlightMode.AUTO:
                self.controller.desired_state(waypoint, vel)

    def cmd_callback(self, msg: TwistStamped) -> None:
        """Parses manual velocity commands from the joystick mapper.

        Args:
            msg (TwistStamped): The manual command twist.
        """
        u_vel = msg.twist.linear.x
        v_vel = msg.twist.linear.y
        w_vel = msg.twist.linear.z
        yaw_cmd = msg.twist.angular.z

        self.manual_cmd[0:4] = [u_vel, v_vel, w_vel, yaw_cmd]

    def fmd_callback(self, msg: FlightMode) -> None:
        """Handles transitions between automatic and manual flight modes.

        Resets integral memories to prevent integral windup jumps during transition.

        Args:
            msg (FlightMode): The incoming flight mode update.
        """
        if self.current_fmd != msg.mode:
            with self.lock:
                self.controller.reset_integrals()

                if msg.mode == FlightMode.AUTO:
                    self.controller.des_position[:] = self.state[9:12]
                    self.controller.des_velocity.fill(0.0)

            self.current_fmd = msg.mode


# =========================================================================
# MAIN FUNCTION
# =========================================================================
def main(args: Optional[List[str]] = None) -> None:
    """Initializes and spins the controller node within a multithreaded executor."""
    rclpy.init(args=args)
    ctrl_node = Controller()

    executor = MultiThreadedExecutor()
    executor.add_node(ctrl_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ctrl_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()