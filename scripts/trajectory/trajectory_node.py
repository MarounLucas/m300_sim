"""ROS 2 node for UAV trajectory execution and FSM management.

This node bridges the mathematical trajectory generation with the ROS
ecosystem, handling waypoint tracking, stabilization, and smooth transitions
between AUTO and MANUAL flight modes using 5th-order polynomial recovery paths.
"""

import os
from typing import List, Tuple

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from m300_msgs.msg import FlightMode
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

# Assuming 'PolinomialTraj' was renamed to 'PolynomialTrajectory' to 
# comply with the English-only rule in the trajectory module.
from .trajectory import PathManager, PolynomialTrajectory

# =========================================================================
# CONSTANTS & CONFIGURATION
# =========================================================================
STATE_IDLE = "IDLE"
STATE_TAKEOFF = "TAKEOFF"
STATE_TRACKING = "TRACKING"
STATE_STABILIZING = "STABILIZING"
STATE_LANDING = "LANDING"
STATE_MANUAL = "MANUAL_MODE"
STATE_RECOVERING = "RECOVERING"


# =========================================================================
# CLASSES
# =========================================================================
class TrajectoryNode(Node):
    """ROS 2 Node for managing UAV trajectories and flight states.

    Attributes:
        path_manager (PathManager): Handles trajectory mathematical generation.
        mission_state (str): Current active state in the FSM.
        curr_position (np.ndarray): Current physical position of the UAV.
    """

    def __init__(self) -> None:
        """Initializes the trajectory node, FSM, and ROS interfaces."""
        super().__init__("trajectory_node")

        # 1. Trajectory Parameter Declaration
        self.declare_parameter(
            "waypoints", [0.0, 0.0, 0.0, 0.0, 0.0, -10.0]
        )
        self.declare_parameter("yaw_mode", "forward")

        wp_flat: List[float] = self.get_parameter("waypoints").value
        yaw_param: str = self.get_parameter("yaw_mode").value

        waypoints = [wp_flat[i : i + 3] for i in range(0, len(wp_flat), 3)]
        self.path_manager = PathManager(waypoints, yaw_mode=yaw_param)
        self.path_manager.generate_path()

        # 2. Unified Mission FSM Configuration
        self.mission_state = STATE_IDLE
        self.traj_time = 0.0
        self.segment_idx = 0

        self.curr_position = np.zeros(3)
        self.idle_pos = np.zeros(3)  # Physical ground parking position

        self.wp_tolerance = 0.5
        self.dt = 1.0 / 100.0

        # 3. Variables for Mode Switching and Recovery
        self.current_fmd = FlightMode.AUTO
        self.paused_pos = np.zeros(3)
        self.paused_yaw = 0.0

        self.recovery_traj = None
        self.recovery_time = 0.0
        self.recovery_duration = 0.0

        # 4. Publishers and Subscribers Initialization
        self.traj_pub = self.create_publisher(
            Odometry, "/m300_sim/trajectory_topic", 10
        )
        self.state_sub = self.create_subscription(
            Odometry, "/m300_sim/telemetry_topic", self.pose_callback, 10
        )
        self.fmd_sub = self.create_subscription(
            FlightMode, "/m300_sim/flight_mode", self.fmd_callback, 10
        )
        self.start_sub = self.create_subscription(
            Empty, "/m300_sim/start_mission", self.start_callback, 10
        )

        # Main trajectory control loop timer
        self.create_timer(self.dt, self.traj_callback)

    def pose_callback(self, msg: Odometry) -> None:
        """Updates the current spatial position of the UAV.

        Args:
            msg (Odometry): The incoming telemetry message.
        """
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.curr_position[0:3] = px, py, pz

    def fmd_callback(self, msg: FlightMode) -> None:
        """Handles transitions between automatic and manual flight modes.

        Args:
            msg (FlightMode): The incoming flight mode update.
        """
        # Transition: AUTO -> MANUAL (Mission Interruption)
        if self.current_fmd == FlightMode.AUTO and msg.mode == FlightMode.MANUAL:
            self.get_logger().info(
                "Interruption detected: Switched to MANUAL mode. "
                "Freezing route."
            )
            self.mission_state = STATE_MANUAL
            state = self.path_manager.get_desired_state(self.traj_time)
            self.paused_pos = state[0]
            self.paused_yaw = state[2]

        # Transition: MANUAL -> AUTO (Mission Resumption)
        elif self.current_fmd == FlightMode.MANUAL and msg.mode == FlightMode.AUTO:
            self.get_logger().info(
                "Resuming mission: Switched to AUTO mode. "
                "Calculating smooth recovery."
            )
            self.mission_state = STATE_RECOVERING
            self.recovery_time = 0.0

            state = self.path_manager.get_desired_state(self.traj_time)
            self.paused_pos = state[0]
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
        """Reads new parameters and initiates a smooth TAKEOFF procedure.

        Args:
            msg (Empty): Empty trigger message.
        """
        if self.mission_state == STATE_IDLE:
            self._reload_parameters()

            if len(self.path_manager.waypoints) > 0:
                target_wp = self.path_manager.waypoints[0][0:3]

                # Generate a slow transition from ground to start point
                dist = np.linalg.norm(self.curr_position - target_wp)
                self.recovery_duration = max(3.0, dist / 2.0)

                self.recovery_traj = PolynomialTrajectory(
                    self.curr_position, target_wp, self.recovery_duration
                )
                self.recovery_traj.generate_path()
                self.recovery_time = 0.0

                self.get_logger().info(
                    f"Initiating smooth TAKEOFF "
                    f"(Estimated duration: {self.recovery_duration:.1f}s)..."
                )
                self.mission_state = STATE_TAKEOFF

    def _reload_parameters(self) -> None:
        """Reloads trajectory configurations from the YAML parameter file."""
        pkg_share = get_package_share_directory("m300_sim")
        ms_yaml = os.path.join(pkg_share, "config", "mission_params.yaml")

        try:
            with open(ms_yaml, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
                ms_data = yaml_data["trajectory_node"]["ros__parameters"]

                wp_flat = ms_data["waypoints"]
                yaw_param = ms_data["yaw_mode"]

                wps = [wp_flat[i : i + 3] for i in range(0, len(wp_flat), 3)]

                if len(wps) > 0:
                    self.path_manager = PathManager(wps, yaw_mode=yaw_param)
                    self.path_manager.generate_path()
                    self.traj_time = 0.0
                    self.segment_idx = 0
        except Exception as err:
            self.get_logger().error(f"Failed to load new waypoints: {err}")

    def _setup_landing(self) -> None:
        """Configures the mathematical trajectory for a safe landing."""
        ground_pos = self.curr_position.copy()
        ground_pos[2] = 0.0  # Set zero altitude exactly below current pos

        dist = np.linalg.norm(self.curr_position - ground_pos)
        self.recovery_duration = max(3.0, dist / 2.0)
        self.recovery_traj = PolynomialTrajectory(
            self.curr_position, ground_pos, self.recovery_duration
        )
        self.recovery_traj.generate_path()

        self.recovery_time = 0.0
        self.mission_state = STATE_LANDING

    def _process_idle(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles the IDLE state.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        return self.idle_pos.copy(), np.zeros(3), 0.0

    def _process_takeoff(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles the TAKEOFF state.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        self.recovery_time += self.dt
        pos = self.recovery_traj.sample_position(self.recovery_time)
        vel = self.recovery_traj.sample_velocity(self.recovery_time)

        if self.recovery_time >= self.recovery_duration:
            self.get_logger().info(
                "TAKEOFF finished. Executing mathematical route..."
            )
            self.traj_time = 0.0
            self.segment_idx = 0
            self.mission_state = STATE_TRACKING

        return pos, vel, 0.0

    def _process_tracking(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles normal route tracking.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        self.traj_time += self.dt
        state = self.path_manager.get_desired_state(self.traj_time)
        pos, vel, yaw = state[0], state[1], state[2]

        start_t = self.path_manager.start_times[self.segment_idx]
        local_time = self.traj_time - start_t

        if local_time >= self.path_manager.segment_times[self.segment_idx]:
            self.get_logger().info(
                f"Stabilizing/Approaching waypoint {self.segment_idx + 1}"
            )
            self.mission_state = STATE_STABILIZING

        return pos, vel, yaw

    def _process_stabilizing(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles waypoint stabilization before proceeding.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        state = self.path_manager.get_desired_state(self.traj_time)
        pos, vel, yaw = state[0], state[1], state[2]

        target_pos = self.path_manager.waypoints[self.segment_idx + 1][0:3]
        error = np.linalg.norm(self.curr_position - target_pos)

        if error < self.wp_tolerance:
            if self.segment_idx < (len(self.path_manager.segments) - 1):
                self.segment_idx += 1
                self.get_logger().info(
                    f"Proceeding to waypoint {self.segment_idx + 2}"
                )
                self.mission_state = STATE_TRACKING
            else:
                self.get_logger().info(
                    "End of route reached. Initiating LANDING..."
                )
                self._setup_landing()

        return pos, vel, yaw

    def _process_landing(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles the automatic landing state.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        self.recovery_time += self.dt
        pos = self.recovery_traj.sample_position(self.recovery_time)
        vel = self.recovery_traj.sample_velocity(self.recovery_time)

        if self.recovery_time >= self.recovery_duration:
            self.get_logger().info(
                "Landing successfully finished. Returning to IDLE state."
            )
            self.idle_pos = self.curr_position.copy()
            self.idle_pos[2] = 0.0  # Ensures ground fixation
            self.mission_state = STATE_IDLE

        return pos, vel, 0.0

    def _process_manual(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles the manual flight mode interruption.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        return self.curr_position.copy(), np.zeros(3), self.paused_yaw

    def _process_recovering(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Handles the recovery transition from MANUAL back to AUTO.

        Returns:
            Tuple: desired position, velocity, and yaw.
        """
        self.recovery_time += self.dt
        pos = self.recovery_traj.sample_position(self.recovery_time)
        vel = self.recovery_traj.sample_velocity(self.recovery_time)
        yaw = self.paused_yaw

        if self.recovery_time >= self.recovery_duration:
            self.get_logger().info(
                "UAV successfully repositioned on route. Resuming mission."
            )
            if self.traj_time >= self.path_manager.total_time:
                self.mission_state = STATE_STABILIZING
            else:
                self.mission_state = STATE_TRACKING

        return pos, vel, yaw

    def _publish_odometry(
        self, pos: np.ndarray, vel: np.ndarray, yaw: float
    ) -> None:
        """Formats and publishes the desired odometry state.

        Args:
            pos (np.ndarray): Target position vector [x, y, z].
            vel (np.ndarray): Target velocity vector [u, v, w].
            yaw (float): Target yaw angle in radians.
        """
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])

        rot = Rotation.from_euler("xyz", [0.0, 0.0, yaw])
        qx, qy, qz, qw = rot.as_quat()
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])

        self.traj_pub.publish(msg)

    def traj_callback(self) -> None:
        """Main control loop triggering the Finite State Machine."""
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


# =========================================================================
# MAIN FUNCTION
# =========================================================================
def main(args=None) -> None:
    """Initializes and spins the Trajectory ROS 2 node."""
    rclpy.init(args=args)
    vec_pub_node = TrajectoryNode()
    try:
        rclpy.spin(vec_pub_node)
    except KeyboardInterrupt:
        pass
    finally:
        vec_pub_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()