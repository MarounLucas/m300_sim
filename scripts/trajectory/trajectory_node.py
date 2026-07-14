"""
ROS 2 node for UAV trajectory execution and finite state machine (FSM) management.

This node bridges the mathematical trajectory generation with the ROS ecosystem,
handling waypoint tracking, stabilization, and smooth transitions between 
AUTO and MANUAL flight modes using 5th-order polynomial recovery paths.
"""

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
import numpy as np

from nav_msgs.msg import Odometry
from m300_msgs.msg import FlightMode

from .trajectory import PathManager, PolinomialTraj

class TrajectoryNode(Node):
    """
    ROS 2 Node that manages the flight mission states and publishes trajectory references.

    Attributes
    ----------
    path_manager : PathManager
        Instance responsible for calculating the mathematical path.
    mission_state : str
        Current state of the FSM ('TRACKING', 'STABILIZING', 'MANUAL_MODE', 'RECOVERING').
    traj_time : float
        Elapsed time in the nominal trajectory execution.
    segment_idx : int
        Index of the current waypoint segment being tracked.
    curr_position : np.ndarray
        The actual spatial position of the UAV [x, y, z] from telemetry.
    current_fmd : int
        The active flight mode (e.g., AUTO or MANUAL).
    """

    def __init__(self) -> None:
        super().__init__('trajectory_node')

        # 1. Trajectory Parameter Declaration
        self.declare_parameter('waypoints', [0.0, 0.0, 0.0, 0.0, 0.0, -10.0])
        self.declare_parameter('yaw_mode', 'forward')

        wp_flat = self.get_parameter('waypoints').value
        yaw_mode_param = self.get_parameter('yaw_mode').value

        waypoints = [wp_flat[i:i+3] for i in range(0, len(wp_flat), 3)]
        self.path_manager = PathManager(waypoints, yaw_mode=yaw_mode_param)
        self.path_manager.generate_path()

        # 2. Unified Mission FSM Configuration
        self.mission_state = "TRACKING"
        self.traj_time = 0.0
        self.segment_idx = 0
        self.curr_position = np.zeros(3)
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
        self.traj_pub = self.create_publisher(Odometry, '/m300_sim/trajectory_topic', 10)
        self.state_sub = self.create_subscription(Odometry, '/m300_sim/telemetry_topic', self.pose_callback, 10)
        self.fmd_sub = self.create_subscription(FlightMode, '/m300_sim/flight_mode', self.fmd_callback, 10)

        # Main trajectory control loop timer
        self.create_timer(self.dt, self.traj_callback)

    def pose_callback(self, msg: Odometry) -> None:
        """
        Updates the UAV's physical position from telemetry data.

        Parameters
        ----------
        msg : Odometry
            The incoming odometry message containing the current pose.
        """
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.curr_position[0:3] = px, py, pz

    def fmd_callback(self, msg: FlightMode) -> None:
        """
        Manages flight mode transitions and trajectory recovery logic.

        Handles the interruption of the mission (AUTO -> MANUAL) by freezing 
        the expected route state, and manages the resumption (MANUAL -> AUTO) 
        by generating a smooth polynomial recovery path.

        Parameters
        ----------
        msg : FlightMode
            The incoming flight mode message.
        """
        # Transition: AUTO -> MANUAL (Mission Interruption)
        if self.current_fmd == FlightMode.AUTO and msg.mode == FlightMode.MANUAL:
            self.get_logger().info('Interruption detected: Switched to MANUAL mode. Freezing route.')
            self.mission_state = "MANUAL_MODE"
            
            # Capture the exact chronological state where the UAV should be
            self.paused_pos, _, self.paused_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            
        # Transition: MANUAL -> AUTO (Mission Resumption)
        elif self.current_fmd == FlightMode.MANUAL and msg.mode == FlightMode.AUTO:
            self.get_logger().info('Resuming mission: Switched to AUTO mode. Calculating smooth recovery.')
            self.mission_state = "RECOVERING"
            self.recovery_time = 0.0
            
            # Ensure the freeze point is updated
            self.paused_pos, _, self.paused_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            
            # Calculate required safe flight time based on PathManager constraints
            self.recovery_duration = self.path_manager.calculate_segment_time(self.curr_position, self.paused_pos)
            
            # Dynamically generate a new polynomial connecting the current manual position to the saved route point
            self.recovery_traj = PolinomialTraj(self.curr_position, self.paused_pos, self.recovery_duration)
            self.recovery_traj.generate_path()
            
        self.current_fmd = msg.mode

    def traj_callback(self) -> None:
        """
        Executes the temporal control logic according to the FSM state.

        Evaluates the current mission state, updates the expected kinematic 
        references (position, velocity, yaw), and publishes them as an 
        Odometry message to be consumed by the inner control loops.
        """
        desire_pos = np.zeros(3)
        desire_vel = np.zeros(3)
        desire_yaw = 0.0

        # --- State 1: Normal Route Tracking ---
        if self.mission_state == "TRACKING":
            self.traj_time += self.dt
            desire_pos, desire_vel, desire_yaw, _ = self.path_manager.get_desired_state(self.traj_time)

            local_time = self.traj_time - self.path_manager.start_times[self.segment_idx]
            if local_time >= self.path_manager.segment_times[self.segment_idx]:
                self.get_logger().info(f'Stabilizing/Approaching waypoint {self.segment_idx + 1}')
                self.mission_state = "STABILIZING"
        
        # --- State 2: Stabilization and Waypoint Verification ---
        elif self.mission_state == "STABILIZING":
            desire_pos, desire_vel, desire_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            target_pos = self.path_manager.waypoints[self.segment_idx + 1][0:3]
            error = np.linalg.norm(self.curr_position - target_pos)

            if error < self.wp_tolerance:
                if self.segment_idx < (len(self.path_manager.segments) - 1):
                    self.segment_idx += 1   
                    self.get_logger().info(f'Proceeding to waypoint {self.segment_idx + 2}')
                    self.mission_state = "TRACKING"
                else:
                    # End of route reached: Maintain fixed hover indefinitely.
                    # Trajectory time stops advancing; desired velocity remains zero.
                    pass

        # --- State 3: Active Manual Mode (Free flight via Joystick) ---
        elif self.mission_state == "MANUAL_MODE":
            # While the pilot is in control, publish the physical position as a 
            # smooth reference to avoid inner loop (CascadeController) jumps.
            desire_pos = self.curr_position.copy()
            desire_vel = np.zeros(3)
            desire_yaw = self.paused_yaw

        # --- State 4: Smooth Trajectory Recovery ---
        elif self.mission_state == "RECOVERING":
            self.recovery_time += self.dt
            
            # Sample the transition polynomial
            desire_pos = self.recovery_traj.sample_position(self.recovery_time)
            desire_vel = self.recovery_traj.sample_velocity(self.recovery_time)
            desire_yaw = self.paused_yaw

            # When the transition curve ends, return to the original chronological route
            if self.recovery_time >= self.recovery_duration:
                self.get_logger().info('UAV successfully repositioned on route. Resuming mission.')
                if self.traj_time >= self.path_manager.total_time:
                    self.mission_state = "STABILIZING"
                else:
                    self.mission_state = "TRACKING"

        # --- Publish Odometry Reference ---
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = float(desire_pos[0])
        msg.pose.pose.position.y = float(desire_pos[1])
        msg.pose.pose.position.z = float(desire_pos[2])

        rot = Rotation.from_euler('xyz', [0.0, 0.0, desire_yaw])
        qx, qy, qz, qw = rot.as_quat()    
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(desire_vel[0])
        msg.twist.twist.linear.y = float(desire_vel[1])
        msg.twist.twist.linear.z = float(desire_vel[2])

        self.traj_pub.publish(msg)


def main(args=None) -> None:
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