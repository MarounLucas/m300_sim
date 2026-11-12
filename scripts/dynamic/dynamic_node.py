"""ROS 2 node for UAV multirotor 6-DoF dynamic simulation.

Executes the 1000 Hz physics loop, mixing virtual control wrenches into rotor
commands, sampling environmental wind models at 100 Hz, optionally injecting
MLP-predicted 6-DoF residual accelerations, and publishing odometry, joint
states, and body-frame accelerations.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import AccelStamped, WrenchStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty

_CURRENT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .mixer import Mixer  # noqa: E402
from .numerical_integration import forward_euler  # noqa: E402
from .quadcopter_model import six_dof_model  # noqa: E402
from .wind_manager import WindManager  # noqa: E402
from neural_network import nn_residual as _nn_residual_mod  # noqa: E402
from neural_network.nn_residual import ResidualNNPredictor  # noqa: E402
import paths  # noqa: E402

DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 1.0 / DEG_TO_RAD


@dataclass
class AircraftModel:
    """Physical and propulsion constants required by the control allocation mixer.

    Attributes:
        mass (float): Total vehicle mass [kg].
        k_t0 (float): Rotor thrust coefficient [N/(rad/s)²].
        k_q0 (float): Rotor torque coefficient [N·m/(rad/s)²].
        dx_arm (float): Longitudinal arm length [m] (body frame).
        dy_fw (float): Front lateral arm length [m] (body frame).
        dy_bw (float): Rear lateral arm length [m] (body frame).
        omega_min (float): Minimum rotor angular speed [rad/s].
        omega_max (float): Maximum rotor angular speed [rad/s].
    """

    mass: float = 0.0
    k_t0: float = 0.0
    k_q0: float = 0.0
    dx_arm: float = 0.0
    dy_fw: float = 0.0
    dy_bw: float = 0.0
    omega_min: float = 0.0
    omega_max: float = 0.0


class DynamicNode(Node):
    """ROS 2 node executing the 6-DoF quadcopter physics simulation at 1000 Hz.

    Attributes:
        ac_model (AircraftModel): Core physical limits and mixer constants.
        ac_params (np.ndarray): 14-element array of physical parameters for ``six_dof_model``.
        state (np.ndarray): Current 20-element 6-DoF state vector.
        mixer (Mixer): Control allocation mixer mapping wrench to motor speeds.
        wind_manager (WindManager): Stochastic and constant wind profile generator.
        nn_predictor (ResidualNNPredictor): Real-time 6-DoF MLP residual predictor.
    """

    def __init__(self) -> None:
        super().__init__('quadcopter_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('mass', 6.47),
                ('jx', 0.214),
                ('jy', 0.200),
                ('jz', 0.389),
                ('jxz', 0.0),
                ('cd', 0.25),
                ('k_t0', 0.0002433),
                ('k_q0', 0.00000692),
                ('tau', 0.015),
                ('kp', 1.0),
                ('xi', 1.0),
                ('dx_arm', 0.335),
                ('dy_fw', 0.365),
                ('dy_bw', 0.315),
                ('omega_min', 60.0),
                ('omega_max', 500.0),
                ('use_nn', False),
            ],
        )

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
        use_nn = bool(self.get_parameter('use_nn').value)

        self.ac_params = np.array(
            [
                mass,
                jx,
                jy,
                jz,
                jxz,
                cd,
                k_t0,
                k_q0,
                tau,
                kp,
                xi,
                dx_arm,
                dy_fw,
                dy_bw,
            ],
            dtype=np.float64,
        )

        self.ac_model = AircraftModel(
            mass=mass,
            k_t0=k_t0,
            k_q0=k_q0,
            dx_arm=dx_arm,
            dy_fw=dy_fw,
            dy_bw=dy_bw,
            omega_min=omega_min,
            omega_max=omega_max,
        )

        self.mixer = Mixer(self.ac_model)
        self.target_state = np.zeros(4)

        hover = mass * 9.81
        self.u_virtual = np.array([hover, 0.0, 0.0, 0.0], dtype=np.float64)

        safe_k_t0 = (
            self.ac_model.k_t0 if self.ac_model.k_t0 > 0.0 else 0.0002433
        )
        omega_hover = float(
            np.sqrt((self.ac_model.mass * 9.81) / (4.0 * safe_k_t0))
        )

        self.init_state = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                omega_hover,
                0.0,
                omega_hover,
                0.0,
                omega_hover,
                0.0,
                omega_hover,
                0.0,
            ],
            dtype=np.float64,
        )
        self.state = self.init_state.copy()

        self.curr_wind = np.zeros(3, dtype=np.float64)
        self.wind_manager = WindManager(
            wind_type='none',
            magnitude=0.0,
            heading=0.0,
            elevation=0.0,
            gust_magnitude=0.0,
        )

        self.nn_predictor = ResidualNNPredictor(enabled=use_nn)
        self.curr_nn_residual = np.zeros(6, dtype=np.float64)

        self.dt = 1.0 / 1000.0
        self.tick_count = 0

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
        self.nn_sub = self.create_subscription(
            Bool, '/m300_sim/nn_enable', self.nn_enable_callback, 10
        )

        self.reload_parameters()
        self.create_timer(self.dt, self.physics_loop)

    def start_callback(self, msg: Empty) -> None:
        """Handle the mission start trigger by reloading YAML parameters."""
        self.reload_parameters()

    def _hot_reload_nn_predictor(self, enabled: bool) -> None:
        """Reload the ``nn_residual`` module and ``.npz`` artifacts from source."""
        try:
            paths.sync_ros_install_scripts()
            importlib.reload(_nn_residual_mod)
            predictor_cls = _nn_residual_mod.ResidualNNPredictor
            if not isinstance(self.nn_predictor, predictor_cls):
                self.nn_predictor = predictor_cls(
                    model_dir=paths.NN_MODELS_DIR, enabled=enabled
                )
            else:
                self.nn_predictor.reload_if_updated()
                self.nn_predictor.set_enabled(enabled)
        except Exception as err:
            self.get_logger().warning(f'NN hot-reload fallback: {err}')
            self.nn_predictor.set_enabled(enabled)

    def nn_enable_callback(self, msg: Bool) -> None:
        """Toggle the 6-DoF neural network residual predictor live from GUI or Xbox."""
        new_state = bool(msg.data)
        if new_state != self.nn_predictor.enabled:
            self._hot_reload_nn_predictor(new_state)
            if not new_state:
                self.curr_nn_residual = np.zeros(6, dtype=np.float64)
            state_str = 'ENABLED' if new_state else 'DISABLED'
            arch_str = 'x'.join(str(d) for d in self.nn_predictor.hidden_dims)
            self.get_logger().info(
                f'Neural Network residual compensation: {state_str} '
                f'(arch={arch_str}, h={self.nn_predictor.h})'
            )

    def reload_parameters(self) -> None:
        """Load and apply physical, wind, and neural network settings from YAML."""
        paths.sync_ros_install_scripts()
        pkg_share = get_package_share_directory('m300_sim')
        ac_yaml = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')
        ms_yaml = os.path.join(pkg_share, 'config', 'mission_params.yaml')

        try:
            with open(ac_yaml, 'r', encoding='utf-8') as file:
                ac_data = yaml.safe_load(file)['quadcopter_node'][
                    'ros__parameters'
                ]

                self.ac_model.mass = ac_data['mass']
                self.ac_model.k_t0 = ac_data['k_t0']
                self.ac_model.k_q0 = ac_data['k_q0']
                self.ac_model.dx_arm = ac_data['dx_arm']
                self.ac_model.dy_fw = ac_data['dy_fw']
                self.ac_model.dy_bw = ac_data['dy_bw']
                self.ac_model.omega_min = ac_data['omega_min']
                self.ac_model.omega_max = ac_data['omega_max']

                self.ac_params = np.array(
                    [
                        ac_data['mass'],
                        ac_data['jx'],
                        ac_data['jy'],
                        ac_data['jz'],
                        ac_data['jxz'],
                        ac_data['cd'],
                        ac_data['k_t0'],
                        ac_data['k_q0'],
                        ac_data['tau'],
                        ac_data['kp'],
                        ac_data['xi'],
                        ac_data['dx_arm'],
                        ac_data['dy_fw'],
                        ac_data['dy_bw'],
                    ],
                    dtype=np.float64,
                )

                self.mixer = Mixer(self.ac_model)

            with open(ms_yaml, 'r', encoding='utf-8') as file:
                ms_data = yaml.safe_load(file)['trajectory_node'][
                    'ros__parameters'
                ]
                w_type_raw = ms_data['wind_type'].lower()
                w_mag = ms_data['wind_magnitude']
                w_head = ms_data['wind_heading'] * DEG_TO_RAD
                w_elev = ms_data['wind_elevation'] * DEG_TO_RAD
                w_gust = ms_data['wind_gust']

                w_type = 'none'
                if 'constant' in w_type_raw:
                    w_type = 'constant'
                elif 'dryden gust' in w_type_raw:
                    w_type = 'dryden'
                elif 'dryden low' in w_type_raw:
                    w_type = 'dryden_lp'
                elif 'sinusoid' in w_type_raw:
                    w_type = 'sinusoid'

                self.wind_manager = WindManager(
                    wind_type=w_type,
                    magnitude=w_mag,
                    heading=w_head,
                    elevation=w_elev,
                    gust_magnitude=w_gust,
                )

                use_nn_flag = bool(
                    ms_data.get('use_nn', self.nn_predictor.enabled)
                )
                self._hot_reload_nn_predictor(use_nn_flag)
                if not self.nn_predictor.enabled:
                    self.curr_nn_residual = np.zeros(6, dtype=np.float64)

        except Exception as err:
            self.get_logger().error(
                f'Hot-Reload synchronization failed: {err}'
            )

    def cmd_callback(self, msg: WrenchStamped) -> None:
        """Store incoming virtual control efforts ``[Fz, Tx, Ty, Tz]``."""
        force_z = msg.wrench.force.z
        torque_x = msg.wrench.torque.x
        torque_y = msg.wrench.torque.y
        torque_z = msg.wrench.torque.z

        self.u_virtual = np.array(
            [force_z, torque_x, torque_y, torque_z], dtype=np.float64
        )

    def physics_loop(self) -> None:
        """Execute one 1000 Hz step of wind, MLP residual inference, and 6-DoF physics."""
        self._update_wind()
        self._step_simulation()
        self._publish_odometry()
        self._log_telemetry()
        self.tick_count += 1

    def _update_wind(self) -> None:
        """Sample the wind disturbance vector at 100 Hz."""
        if self.tick_count % 10 == 0:
            t_sim = self.tick_count * self.dt
            current_pos = self.state[9:12]
            self.curr_wind = self.wind_manager.get_wind(t_sim, current_pos)

    def _step_simulation(self) -> None:
        """Advance the 6-DoF physics model by one time step (`dt = 0.001 s`)."""
        w_cmds = self.mixer.compute_motor_speed(self.u_virtual)
        if self.tick_count % 20 == 0:
            self._publish_motor_speeds(w_cmds)

        if self.tick_count % 10 == 0:
            self.curr_nn_residual = self.nn_predictor.step(
                self.state, self.u_virtual, self.curr_wind
            )

        nn_vec = (
            self.curr_nn_residual if self.nn_predictor.enabled else None
        )
        dx = six_dof_model(
            self.state, self.ac_params, self.curr_wind, w_cmds, nn_vec
        )
        self.state = forward_euler(self.state, dx, self.dt)

        if self.tick_count % 10 == 0:
            self._publish_acceleration(dx)

    def _publish_motor_speeds(self, w_cmds: np.ndarray) -> None:
        """Publish the current angular speeds of all four rotors."""
        if not rclpy.ok():
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['rotor_0', 'rotor_1', 'rotor_2', 'rotor_3']
        msg.velocity = [
            float(w_cmds[0]),
            float(w_cmds[1]),
            float(w_cmds[2]),
            float(w_cmds[3]),
        ]
        if rclpy.ok():
            self.motor_pub.publish(msg)

    def _publish_odometry(self) -> None:
        """Construct and publish the current 6-DoF state as a ROS 2 Odometry message."""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = float(self.state[9])
        msg.pose.pose.position.y = float(self.state[10])
        msg.pose.pose.position.z = float(self.state[11])

        rot = Rotation.from_euler(
            'xyz', [self.state[6], self.state[7], self.state[8]]
        )
        qx, qy, qz, qw = rot.as_quat()

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(self.state[0])
        msg.twist.twist.linear.y = float(self.state[1])
        msg.twist.twist.linear.z = float(self.state[2])

        msg.twist.twist.angular.x = float(self.state[3])
        msg.twist.twist.angular.y = float(self.state[4])
        msg.twist.twist.angular.z = float(self.state[5])

        if rclpy.ok():
            self.dyn_pub.publish(msg)

    def _publish_acceleration(self, dx: np.ndarray) -> None:
        """Publish body-frame linear and angular accelerations."""
        if not rclpy.ok():
            return
        msg = AccelStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.accel.linear.x = float(dx[0])
        msg.accel.linear.y = float(dx[1])
        msg.accel.linear.z = float(dx[2])

        msg.accel.angular.x = float(dx[3])
        msg.accel.angular.y = float(dx[4])
        msg.accel.angular.z = float(dx[5])

        if rclpy.ok():
            self.accel_pub.publish(msg)

    def _log_telemetry(self) -> None:
        """Log a human-readable telemetry summary to the console at 4 Hz."""
        if self.tick_count % 250 == 0:
            rot = Rotation.from_euler(
                'xyz', [self.state[6], self.state[7], self.state[8]]
            )
            roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
            nn_status = 'ON' if self.nn_predictor.enabled else 'OFF'

            self.get_logger().info(
                f'--- M350 Telemetry (NN: {nn_status}) ---\n'
                f'Pos (m): X={self.state[9]:.2f}, Y={self.state[10]:.2f}, '
                f'Z={self.state[11]:.2f}\n'
                f'Vel (m/s): u={self.state[0]:.2f}, v={self.state[1]:.2f}, '
                f'w={self.state[2]:.2f}\n'
                f'Attitude (deg): Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}\n'
            )


def main(args: list[str] | None = None) -> None:
    """Initialize and spin the dynamic simulation ROS 2 node."""
    rclpy.init(args=args)
    node = DynamicNode()

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
