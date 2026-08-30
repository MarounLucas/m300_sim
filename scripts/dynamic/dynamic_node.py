"""ROS 2 node for UAV multirotor dynamic simulation.

This module handles the physics engine, calculating the drone's states based
on motor thrust, environmental wind, and rigid-body 6-DOF models.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import WrenchStamped, AccelStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

from .mixer import Mixer
from .numerical_integration import forward_euler
from .quadcopter_model import six_dof_model
from .wind_manager import WindManager

# =========================================================================
# CONSTANTS & CONFIGURATION
# =========================================================================
DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 1.0 / DEG_TO_RAD


# =========================================================================
# CLASSES
# =========================================================================
@dataclass
class AircraftModel:
    """Data structure representing the physical parameters of the aircraft."""
    mass: float = 0.0
    k_t0: float = 0.0
    k_q0: float = 0.0
    dx_arm: float = 0.0
    dy_fw: float = 0.0
    dy_bw: float = 0.0
    omega_min: float = 0.0
    omega_max: float = 0.0


class DynamicNode(Node):
    """ROS 2 Node executing the 6-DOF physics simulation of the UAV.

    Attributes:
        ac_model (AircraftModel): Object holding core physical limits and constants.
        ac_params (np.ndarray): Flattened array of aerodynamic and physical parameters.
        state (np.ndarray): The current 6-DOF state vector of the UAV.
        mixer (Mixer): Component that maps control wrenches to motor speeds.
        wind_manager (WindManager): Component that injects wind disturbances.
    """

    def __init__(self) -> None:
        """Initializes the dynamic simulation node, parameters, and ROS interfaces."""
        super().__init__("quadcopter_node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('mass', 6.47), ('jx', 0.214), ('jy', 0.200), ('jz', 0.389),
                ('jxz', 0.0), ('cd', 0.25), ('k_t0', 0.0002433), ('k_q0', 0.00000692),
                ('tau', 0.015), ('kp', 1.0), ('xi', 1.0), ('dx_arm', 0.335),
                ('dy_fw', 0.365), ('dy_bw', 0.315), ('omega_min', 60.0),
                ('omega_max', 500.0)
            ]
        )

        # Retrieve values loaded via YAML
        mass = float(self.get_parameter('mass').value)
        jx = float(self.get_parameter('jx').value)
        jy = float(self.get_parameter('jy').value)
        jz = float(self.get_parameter('jz').value)
        jxz = float(self.get_parameter('jxz').value)
        cd = float(self.get_parameter('cd').value)
        k_t0 = float(self.get_parameter('k_t0').value)
        k_q0 = float(self.get_parameter('k_q0').value)
        tau = float(self.get_parameter('tau').value)
        kp = float(self.get_parameter('kp').value)
        xi = float(self.get_parameter('xi').value)
        dx_arm = float(self.get_parameter('dx_arm').value)
        dy_fw = float(self.get_parameter('dy_fw').value)
        dy_bw = float(self.get_parameter('dy_bw').value)
        omega_min = float(self.get_parameter('omega_min').value)
        omega_max = float(self.get_parameter('omega_max').value)

        self.ac_params = np.array([
            mass, jx, jy, jz, jxz,
            cd, k_t0, k_q0,
            tau, kp, xi,
            dx_arm, dy_fw, dy_bw,
        ], dtype=np.float64)

        self.ac_model = AircraftModel(
            mass=mass, k_t0=k_t0, k_q0=k_q0, dx_arm=dx_arm,
            dy_fw=dy_fw, dy_bw=dy_bw, omega_min=omega_min, omega_max=omega_max
        )

        # Initialize Mixer
        self.mixer = Mixer(self.ac_model)

        # X, Y, Z, Yaw
        self.target_state = np.zeros(4)

        # Initializing engines (Virtual control efforts)
        self.u_virtual = np.zeros(4)

        # FIX: Protection against division by zero in case the YAML file has zeros.
        safe_k_t0 = self.ac_model.k_t0 if self.ac_model.k_t0 > 0.0 else 0.0002433
        omega_hover = np.sqrt((self.ac_model.mass * 9.81) / (4.0 * safe_k_t0))

        # Initial 6-DOF state vector
        self.init_state = np.array([
            0.0, 0.0, 0.0,                                      # u, v, w
            0.0, 0.0, 0.0,                                      # p, q, r
            0.0 * DEG_TO_RAD, 0.0 * DEG_TO_RAD, 0.0 * DEG_TO_RAD, # phi, theta, psi
            0.0, 0.0, 0.0,                                      # px, py, pz
            omega_hover, 0.0,                                   # w_m1, alpha_m1
            omega_hover, 0.0,                                   # w_m2, alpha_m2
            omega_hover, 0.0,                                   # w_m3, alpha_m3
            omega_hover, 0.0                                    # w_m4, alpha_m4
        ])
        self.state = self.init_state.copy()

        self.curr_wind = np.zeros(3)
        self.wind_manager = WindManager(
            wind_type="none", magnitude=0.0, heading=0.0,
            elevation=0.0, gust_magnitude=0.0
        )

        self.dt = 1.0 / 1000.0
        self.tick_count = 0

        # ROS 2 Interfaces
        self.dyn_pub = self.create_publisher(
            Odometry, '/m300_sim/telemetry_topic', 10
        )
        
        self.motor_pub = self.create_publisher(
            JointState, '/m300_sim/joint_states', 10
        )        

        self.accel_pub = self.create_publisher(
            AccelStamped, '/m300_sim/acceleration_topic', 10
        )

        self.ctrl_sub = self.create_subscription(
            WrenchStamped, '/m300_sim/control_topic', self.cmd_callback, 10
        )

        self.start_sub = self.create_subscription(
            Empty, '/m300_sim/start_mission', self.start_callback, 10
        )

        self.reload_parameters()
        self.create_timer(self.dt, self.physics_loop)

    def start_callback(self, msg: Empty) -> None:
        """Handles the mission start signal by reloading parameters.

        Args:
            msg (Empty): The empty trigger message.
        """
        self.reload_parameters()

    def reload_parameters(self) -> None:
        """Loads and applies physical and environmental parameters from YAML files."""
        pkg_share = get_package_share_directory('m300_sim')
        ac_yaml = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')
        ms_yaml = os.path.join(pkg_share, 'config', 'mission_params.yaml')

        try:
            # 1. Load Aircraft physical parameters
            with open(ac_yaml, 'r', encoding='utf-8') as file:
                ac_data = yaml.safe_load(file)['quadcopter_node']['ros__parameters']
                
                self.ac_model.mass = ac_data['mass']
                self.ac_model.k_t0 = ac_data['k_t0']
                self.ac_model.k_q0 = ac_data['k_q0']
                self.ac_model.dx_arm = ac_data['dx_arm']
                self.ac_model.dy_fw = ac_data['dy_fw']
                self.ac_model.dy_bw = ac_data['dy_bw']
                self.ac_model.omega_min = ac_data['omega_min']
                self.ac_model.omega_max = ac_data['omega_max']

                self.ac_params = np.array([
                    ac_data['mass'], ac_data['jx'], ac_data['jy'], ac_data['jz'],
                    ac_data['jxz'], ac_data['cd'], ac_data['k_t0'], ac_data['k_q0'],
                    ac_data['tau'], ac_data['kp'], ac_data['xi'],
                    ac_data['dx_arm'], ac_data['dy_fw'], ac_data['dy_bw'],
                ], dtype=np.float64)

                # Recalculate propeller thrust equations
                self.mixer = Mixer(self.ac_model)

            # 2. Load Mission Climate and Wind parameters
            with open(ms_yaml, 'r', encoding='utf-8') as file:
                ms_data = yaml.safe_load(file)['trajectory_node']['ros__parameters']
                w_type_raw = ms_data['wind_type'].lower()
                w_mag = ms_data['wind_magnitude']
                w_head = ms_data['wind_heading'] * DEG_TO_RAD
                w_elev = ms_data['wind_elevation'] * DEG_TO_RAD
                w_gust = ms_data['wind_gust']

                # Translate UI strings to WindManager identifiers
                w_type = "none"
                if "constant" in w_type_raw:
                    w_type = "constant"
                elif "dryden gust" in w_type_raw:
                    w_type = "dryden"
                elif "dryden low" in w_type_raw:
                    w_type = "dryden_lp"
                elif "sinusoid" in w_type_raw:
                    w_type = "sinusoidal"

                self.wind_manager = WindManager(
                    wind_type=w_type,
                    magnitude=w_mag,
                    heading=w_head,
                    elevation=w_elev,
                    gust_magnitude=w_gust
                )

        except Exception as err:
            self.get_logger().error(f"Hot-Reload synchronization failed: {err}")

    def cmd_callback(self, msg: WrenchStamped) -> None:
        """Stores the incoming control efforts (forces and torques).

        Args:
            msg (WrenchStamped): The requested control efforts.
        """
        force_z = msg.wrench.force.z
        torque_x = msg.wrench.torque.x
        torque_y = msg.wrench.torque.y
        torque_z = msg.wrench.torque.z

        self.u_virtual = np.array([force_z, torque_x, torque_y, torque_z])

    def physics_loop(self) -> None:
        """Main timer callback executing the dynamic simulation at 1000Hz."""
        self._update_wind()
        self._step_simulation()
        self._publish_odometry()
        self._log_telemetry()
        self.tick_count += 1

    def _update_wind(self) -> None:
        """Updates the wind vector at a 100Hz frequency."""
        if self.tick_count % 10 == 0:
            t_sim = self.tick_count * self.dt
            current_pos = self.state[9:12]
            self.curr_wind = self.wind_manager.get_wind(t_sim, current_pos)

    def _step_simulation(self) -> None:
        """Advances the 6-DOF physics model by one time step."""
        w_cmds = self.mixer.compute_motor_speed(self.u_virtual)
        
        self._publish_motor_speeds(w_cmds)
        
        dx = six_dof_model(self.state, self.ac_params, self.curr_wind, w_cmds)
        self.state = forward_euler(self.state, dx, self.dt)

        self._publish_acceleration(dx)

    def _publish_motor_speeds(self, w_cmds: np.ndarray) -> None:
        """Publishes the current rotation speed of all 4 propellers using JointState."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['rotor_0', 'rotor_1', 'rotor_2', 'rotor_3']
        msg.velocity = [float(w_cmds[0]), float(w_cmds[1]), float(w_cmds[2]), float(w_cmds[3])]
        self.motor_pub.publish(msg)

    def _publish_odometry(self) -> None:
        """Constructs and publishes the Odometry message with the current state."""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        # Position
        msg.pose.pose.position.x = float(self.state[9])
        msg.pose.pose.position.y = float(self.state[10])
        msg.pose.pose.position.z = float(self.state[11])

        # Orientation
        rot = Rotation.from_euler(
            "xyz", [self.state[6], self.state[7], self.state[8]]
        )
        qx, qy, qz, qw = rot.as_quat()
        
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Linear Velocity
        msg.twist.twist.linear.x = self.state[0]
        msg.twist.twist.linear.y = self.state[1]
        msg.twist.twist.linear.z = self.state[2]

        # Angular Velocity
        msg.twist.twist.angular.x = self.state[3]
        msg.twist.twist.angular.y = self.state[4]
        msg.twist.twist.angular.z = self.state[5]

        self.dyn_pub.publish(msg)

    def _publish_acceleration(self, dx: np.ndarray):
        '''Constructs and publishes the analytical accelerations'''

        msg = AccelStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.accel.linear.x = float(dx[0])
        msg.accel.linear.y = float(dx[1])
        msg.accel.linear.z = float(dx[2])

        msg.accel.angular.x = float(dx[3])
        msg.accel.angular.y = float(dx[4])
        msg.accel.angular.z = float(dx[5])

        self.accel_pub.publish(msg)


    def _log_telemetry(self) -> None:
        """Logs a human-readable telemetry summary to the console at 10Hz."""
        if self.tick_count % 100 == 0:
            rot = Rotation.from_euler(
                "xyz", [self.state[6], self.state[7], self.state[8]]
            )
            roll, pitch, yaw = rot.as_euler('xyz', degrees=True) 

            self.get_logger().info(
                f'--- M350 Telemetry ---\n'
                f'Pos (m): X={self.state[9]:.2f}, Y={self.state[10]:.2f}, '
                f'Z={self.state[11]:.2f}\n'
                f'Vel (m/s): u={self.state[0]:.2f}, v={self.state[1]:.2f}, '
                f'w={self.state[2]:.2f}\n'
                f'Attitude (deg): Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}\n'
            )


# =========================================================================
# MAIN FUNCTION
# =========================================================================
def main(args: Optional[List[str]] = None) -> None:
    """Initializes and spins the dynamic simulation ROS 2 node."""
    rclpy.init(args=args)
    node = DynamicNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()