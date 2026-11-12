"""ROS 2 node for UAV trajectory execution and flight state machine management.

Bridges mathematical trajectory generation with the ROS 2 ecosystem, handling
waypoint tracking, waypoint stabilization pauses, and smooth 5th-order
polynomial transitions between AUTO and MANUAL flight modes. Synchronized at
100 Hz by downsampling the 1000 Hz physics telemetry stream.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ament_index_python.packages import get_package_share_directory
from m300_msgs.msg import FlightMode
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty

from .trajectory import PathManager, PolynomialTrajectory

STATE_IDLE = 'IDLE'
STATE_TAKEOFF = 'TAKEOFF'
STATE_TRACKING = 'TRACKING'
STATE_STABILIZING = 'STABILIZING'
STATE_LANDING = 'LANDING'
STATE_MANUAL = 'MANUAL_MODE'
STATE_RECOVERING = 'RECOVERING'


class TrajectoryNode(Node):
    """Manage UAV trajectory generation and mission finite state machine.

    Attributes:
        path_manager (PathManager): Mathematical trajectory and yaw generator.
        mission_state (str): Active state in the mission FSM.
        curr_position (np.ndarray): Current UAV position [x, y, z] [m] (NED frame).
    """

    def __init__(self) -> None:
        """Initialize the trajectory node, mission FSM, and ROS 2 interfaces."""
        super().__init__('trajectory_node')

        self.declare_parameter(
            'waypoints', [0.0, 0.0, 0.0, 0.0, 0.0, -10.0]
        )
        self.declare_parameter('yaw_mode', 'forward')
        self.declare_parameter('yaw_target', [0.0, 0.0])
        self.declare_parameter('max_speed_xy', 9.2)
        self.declare_parameter('max_speed_z', 2.0)

        wp_flat: list[float] = self.get_parameter('waypoints').value
        yaw_param: str = self.get_parameter('yaw_mode').value
        yaw_target_param: list[float] = self.get_parameter('yaw_target').value
        speed_xy = float(self.get_parameter('max_speed_xy').value)
        speed_z = float(self.get_parameter('max_speed_z').value)

        waypoints = [wp_flat[i : i + 3] for i in range(0, len(wp_flat), 3)]
        self.path_manager = PathManager(
            waypoints,
            yaw_mode=yaw_param,
            yaw_target=yaw_target_param,
            xy_velocity=speed_xy,
            z_velocity=speed_z,
        )
        self.path_manager.generate_path()

        self.mission_state = STATE_IDLE
        self.traj_time = 0.0
        self.segment_idx = 0
        self.tick = 0

        self.curr_position = np.zeros(3)
        self.idle_pos = np.zeros(3)

        self.wp_tolerance = 0.15
        self.dt = 1.0 / 100.0
        self.stabilize_time = 0.0

        self.current_fmd = FlightMode.AUTO
        self.paused_pos = np.zeros(3)
        self.paused_yaw = 0.0

        self.recovery_traj: PolynomialTrajectory | None = None
        self.recovery_time = 0.0
        self.recovery_duration = 0.0

        self.traj_pub = self.create_publisher(
            Odometry, '/m300_sim/trajectory_topic', 10
        )
        self.state_sub = self.create_subscription(
            Odometry, '/m300_sim/telemetry_topic', self.pose_callback, 10
        )
        self.fmd_sub = self.create_subscription(
            FlightMode, '/m300_sim/flight_mode', self.fmd_callback, 10
        )
        self.start_sub = self.create_subscription(
            Empty, '/m300_sim/start_mission', self.start_callback, 10
        )

    def pose_callback(self, msg: Odometry) -> None:
        """Update the UAV position from telemetry and step the FSM at 100 Hz.

        Downsamples the 1000 Hz physics telemetry by a factor of 10 so the
        trajectory state machine stays locked to simulation time.

        Args:
            msg: Incoming UAV odometry telemetry message.
        """
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.curr_position[0:3] = px, py, pz

        self.tick += 1
        if self.tick % 10 == 0:
            self.traj_callback()

    def fmd_callback(self, msg: FlightMode) -> None:
        """Handle transitions between automatic and manual flight modes.

        Args:
            msg: Incoming flight mode message.
        """
        if self.current_fmd == FlightMode.AUTO and msg.mode == FlightMode.MANUAL:
            self.get_logger().info(
                'Interruption detected: Switched to MANUAL mode. '
                'Freezing route.'
            )
            self.mission_state = STATE_MANUAL
            state = self.path_manager.get_desired_state(self.traj_time)
            self.paused_pos = state[0].copy()
            self.paused_yaw = state[2]

        elif self.current_fmd == FlightMode.MANUAL and msg.mode == FlightMode.AUTO:
            self.get_logger().info(
                'Resuming mission: Switched to AUTO mode. '
                'Calculating smooth recovery.'
            )
            self.mission_state = STATE_RECOVERING
            self.recovery_time = 0.0

            state = self.path_manager.get_desired_state(self.traj_time)
            self.paused_pos = state[0].copy()
            self.paused_yaw = state[2]

            calc_time = self.path_manager.calculate_segment_time(
                self.curr_position, self.paused_pos
            )
            self.recovery_duration = max(3.0, calc_time)

            self.recovery_traj = PolynomialTrajectory(
                self.curr_position, self.paused_pos, self.recovery_duration
            )
            self.recovery_traj.generate_path()

        self.current_fmd = msg.mode

    def start_callback(self, msg: Empty) -> None:
        """Reload mission parameters and initiate a smooth takeoff trajectory.

        Args:
            msg: Empty trigger message from the GUI.
        """
        if self.mission_state == STATE_IDLE:
            self._reload_parameters()

            if len(self.path_manager.waypoints) > 0:
                target_wp = self.path_manager.waypoints[0][0:3]

                dist = float(np.linalg.norm(self.curr_position - target_wp))
                self.recovery_duration = max(3.0, dist / 2.0)

                self.recovery_traj = PolynomialTrajectory(
                    self.curr_position, target_wp, self.recovery_duration
                )
                self.recovery_traj.generate_path()
                self.recovery_time = 0.0

                self.get_logger().info(
                    f'Initiating smooth TAKEOFF '
                    f'(Estimated duration: {self.recovery_duration:.1f}s)...'
                )
                self.mission_state = STATE_TAKEOFF

    def _reload_parameters(self) -> None:
        """Reload trajectory configuration from ``mission_params.yaml``."""
        pkg_share = get_package_share_directory('m300_sim')
        ms_yaml = os.path.join(pkg_share, 'config', 'mission_params.yaml')

        try:
            with open(ms_yaml, 'r', encoding='utf-8') as file:
                yaml_data = yaml.safe_load(file)
                ms_data = yaml_data['trajectory_node']['ros__parameters']

                wp_flat = ms_data['waypoints']
                yaw_param = ms_data.get('yaw_mode', 'forward')
                yaw_target_param = ms_data.get('yaw_target', [0.0, 0.0])
                speed_xy = ms_data.get('max_speed_xy', 9.2)
                speed_z = ms_data.get('max_speed_z', 2.0)

                wps = [wp_flat[i : i + 3] for i in range(0, len(wp_flat), 3)]

                if len(wps) > 0:
                    self.path_manager = PathManager(
                        wps,
                        yaw_mode=yaw_param,
                        yaw_target=yaw_target_param,
                        xy_velocity=speed_xy,
                        z_velocity=speed_z,
                    )
                    self.path_manager.generate_path()
                    self.traj_time = 0.0
                    self.segment_idx = 0
        except Exception as err:
            self.get_logger().error(f'Failed to load new waypoints: {err}')

    def _setup_landing(self) -> None:
        """Configure the vertical descent polynomial trajectory to ground level."""
        if abs(self.curr_position[2]) < 0.25:
            self.get_logger().info(
                'Already at ground level. Returning to IDLE state.'
            )
            self.idle_pos = self.curr_position.copy()
            self.idle_pos[2] = 0.0
            self.mission_state = STATE_IDLE
            return

        target_wp = self.path_manager.waypoints[-1]
        start_pos = np.array([
            target_wp[0], target_wp[1], self.curr_position[2]
        ])
        ground_pos = np.array([target_wp[0], target_wp[1], 0.0])

        dist = float(np.linalg.norm(start_pos - ground_pos))
        self.recovery_duration = max(4.0, dist / 1.5)
        self.recovery_traj = PolynomialTrajectory(
            start_pos, ground_pos, self.recovery_duration
        )
        self.recovery_traj.generate_path()

        self.recovery_time = 0.0
        self.mission_state = STATE_LANDING

    def _process_idle(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ground-hold references for the IDLE state."""
        return self.idle_pos.copy(), np.zeros(3), 0.0

    def _process_takeoff(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample the takeoff polynomial and transition to route tracking."""
        self.recovery_time += self.dt
        assert self.recovery_traj is not None
        pos = self.recovery_traj.sample_position(self.recovery_time)
        vel = self.recovery_traj.sample_velocity(self.recovery_time)

        yaw = (
            self.path_manager.calculate_yaw(0, 0.0, pos)
            if self.path_manager.yaw_mode == 'target'
            else 0.0
        )

        if self.recovery_time >= self.recovery_duration:
            self.get_logger().info(
                'TAKEOFF finished. Executing mathematical route...'
            )
            self.traj_time = 0.0
            self.segment_idx = 0
            self.mission_state = STATE_TRACKING

        return pos, vel, yaw

    def _process_tracking(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample the active polynomial segment and advance to stabilization at its end."""
        self.traj_time += self.dt
        start_t = self.path_manager.start_times[self.segment_idx]
        seg_duration = self.path_manager.segment_times[self.segment_idx]
        local_time = self.traj_time - start_t

        # Clamp sample time to the current segment end so yaw is not sampled from
        # the next segment before waypoint stabilization completes.
        sample_t = min(self.traj_time, start_t + seg_duration - 1e-6)
        state = self.path_manager.get_desired_state(sample_t)
        pos, vel, yaw = state[0], state[1], state[2]

        if local_time >= seg_duration:
            self.get_logger().info(
                f'Stabilizing/Approaching waypoint {self.segment_idx + 1}'
            )
            self.mission_state = STATE_STABILIZING
            self.stabilize_time = 0.0

        return pos, vel, yaw

    def _process_stabilizing(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Hold the target waypoint reference and require 2 s within tolerance."""
        target_pos = self.path_manager.waypoints[self.segment_idx + 1][0:3].copy()
        vel = np.zeros(3)
        yaw = (
            self.path_manager.calculate_yaw(self.segment_idx, 0.0, target_pos)
            if self.path_manager.yaw_mode == 'target'
            else self.path_manager.last_yaw
        )

        error_pos = float(np.linalg.norm(self.curr_position - target_pos))

        if error_pos < self.wp_tolerance:
            self.stabilize_time += self.dt

            if self.stabilize_time >= 2.0:
                if self.segment_idx < (len(self.path_manager.segments) - 1):
                    self.segment_idx += 1
                    self.traj_time = self.path_manager.start_times[self.segment_idx]
                    self.get_logger().info(
                        f'Proceeding to waypoint {self.segment_idx + 2}'
                    )
                    self.mission_state = STATE_TRACKING
                else:
                    self.get_logger().info(
                        'End of route reached. Initiating LANDING...'
                    )
                    self._setup_landing()
        else:
            # Reset timer if wind pushes the UAV outside the tolerance sphere
            self.stabilize_time = 0.0

        return target_pos, vel, yaw

    def _process_landing(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample the vertical landing trajectory until touchdown."""
        self.recovery_time += self.dt
        assert self.recovery_traj is not None
        pos = self.recovery_traj.sample_position(self.recovery_time)
        vel = self.recovery_traj.sample_velocity(self.recovery_time)

        if self.recovery_time >= self.recovery_duration:
            self.get_logger().info(
                'Landing successfully finished. Returning to IDLE state.'
            )
            self.idle_pos = self.curr_position.copy()
            self.idle_pos[2] = 0.0
            self.mission_state = STATE_IDLE

        return pos, vel, 0.0

    def _process_manual(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Track the current UAV position while manual control is active."""
        return self.curr_position.copy(), np.zeros(3), self.paused_yaw

    def _process_recovering(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample the smooth recovery trajectory from manual position back to the route."""
        self.recovery_time += self.dt
        assert self.recovery_traj is not None
        pos = self.recovery_traj.sample_position(self.recovery_time)
        vel = self.recovery_traj.sample_velocity(self.recovery_time)
        yaw = self.paused_yaw

        if self.recovery_time >= self.recovery_duration:
            self.get_logger().info(
                'UAV successfully repositioned on route. Resuming mission.'
            )
            if self.traj_time >= self.path_manager.total_time:
                self.mission_state = STATE_STABILIZING
            else:
                self.mission_state = STATE_TRACKING

        return pos, vel, yaw

    def _publish_odometry(
        self, pos: np.ndarray, vel: np.ndarray, yaw: float
    ) -> None:
        """Publish the desired position, velocity, and yaw as an Odometry message.

        Args:
            pos: Target position vector [x, y, z] [m] (NED frame).
            vel: Target velocity vector [vx, vy, vz] [m/s] (NED frame).
            yaw: Target yaw angle [rad] (NED frame).
        """
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])

        rot = Rotation.from_euler('xyz', [0.0, 0.0, yaw])
        qx, qy, qz, qw = rot.as_quat()
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])

        if rclpy.ok():
            self.traj_pub.publish(msg)

    def traj_callback(self) -> None:
        """Step the active mission state and publish the resulting trajectory reference."""
        if self.mission_state == STATE_IDLE:
            pos, vel, yaw = self._process_idle()
        elif self.mission_state == STATE_TAKEOFF:
            pos, vel, yaw = self._process_takeoff()
        elif self.mission_state == STATE_TRACKING:
            pos, vel, yaw = self._process_tracking()
        elif self.mission_state == STATE_STABILIZING:
            pos, vel, yaw = self._process_stabilizing()
        elif self.mission_state == STATE_LANDING:
            pos, vel, yaw = self._process_landing()
        elif self.mission_state == STATE_MANUAL:
            pos, vel, yaw = self._process_manual()
        elif self.mission_state == STATE_RECOVERING:
            pos, vel, yaw = self._process_recovering()
        else:
            pos, vel, yaw = np.zeros(3), np.zeros(3), 0.0

        self._publish_odometry(pos, vel, yaw)


def main(args: list[str] | None = None) -> None:
    """Initialize and spin the trajectory ROS 2 node."""
    rclpy.init(args=args)
    vec_pub_node = TrajectoryNode()
    try:
        rclpy.spin(vec_pub_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            vec_pub_node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
