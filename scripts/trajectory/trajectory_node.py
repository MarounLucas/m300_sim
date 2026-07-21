"""
ROS 2 node for UAV trajectory execution and finite state machine (FSM) management.

This node bridges the mathematical trajectory generation with the ROS ecosystem,
handling waypoint tracking, stabilization, and smooth transitions between 
AUTO and MANUAL flight modes using 5th-order polynomial recovery paths.
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
import numpy as np

from nav_msgs.msg import Odometry
from m300_msgs.msg import FlightMode
from std_msgs.msg import Empty

from .trajectory import PathManager, PolinomialTraj

class TrajectoryNode(Node):
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
        self.mission_state = "IDLE"
        self.traj_time = 0.0
        self.segment_idx = 0
        
        self.curr_position = np.zeros(3)
        self.idle_pos = np.zeros(3)  # Posição onde o drone está fisicamente estacionado no chão
        
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
        self.start_sub = self.create_subscription(Empty, '/m300_sim/start_mission', self.start_callback, 10)

        # Main trajectory control loop timer
        self.create_timer(self.dt, self.traj_callback)

    def pose_callback(self, msg: Odometry) -> None:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.curr_position[0:3] = px, py, pz

    def fmd_callback(self, msg: FlightMode) -> None:
        # Transition: AUTO -> MANUAL (Mission Interruption)
        if self.current_fmd == FlightMode.AUTO and msg.mode == FlightMode.MANUAL:
            self.get_logger().info('Interruption detected: Switched to MANUAL mode. Freezing route.')
            self.mission_state = "MANUAL_MODE"
            self.paused_pos, _, self.paused_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            
        # Transition: MANUAL -> AUTO (Mission Resumption)
        elif self.current_fmd == FlightMode.MANUAL and msg.mode == FlightMode.AUTO:
            self.get_logger().info('Resuming mission: Switched to AUTO mode. Calculating smooth recovery.')
            self.mission_state = "RECOVERING"
            self.recovery_time = 0.0
            
            self.paused_pos, _, self.paused_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            calc_time = self.path_manager.calculate_segment_time(self.curr_position, self.paused_pos)
            self.recovery_duration = max(3.0, calc_time) 
            
            self.recovery_traj = PolinomialTraj(self.curr_position, self.paused_pos, self.recovery_duration)
            self.recovery_traj.generate_path()
            
        self.current_fmd = msg.mode

    def start_callback(self, msg: Empty):
        """Lê os parâmetros novos e inicia o procedimento de TAKEOFF suave."""
        if self.mission_state == "IDLE":
            self.reload_parameters()
            
            if len(self.path_manager.waypoints) > 0:
                target_wp = self.path_manager.waypoints[0][0:3]
                
                # Gera uma transição lenta do solo até o ponto inicial da missão (2 m/s de média, mínimo 3s)
                dist = np.linalg.norm(self.curr_position - target_wp)
                self.recovery_duration = max(3.0, dist / 2.0)
                
                self.recovery_traj = PolinomialTraj(self.curr_position, target_wp, self.recovery_duration)
                self.recovery_traj.generate_path()
                self.recovery_time = 0.0
                
                self.get_logger().info(f'Iniciando TAKEOFF suave (Duração estimada: {self.recovery_duration:.1f}s)...')
                self.mission_state = "TAKEOFF"

    def reload_parameters(self):
        pkg_share = get_package_share_directory('m300_sim')
        ms_yaml = os.path.join(pkg_share, 'config', 'mission_params.yaml')
        
        try:
            with open(ms_yaml, 'r') as f:
                ms_data = yaml.safe_load(f)['trajectory_node']['ros__parameters']
                wp_flat = ms_data['waypoints']
                yaw_mode_param = ms_data['yaw_mode']
                
                waypoints = [wp_flat[i:i+3] for i in range(0, len(wp_flat), 3)]
                
                if len(waypoints) > 0:
                    self.path_manager = PathManager(waypoints, yaw_mode=yaw_mode_param)
                    self.path_manager.generate_path()
                    self.traj_time = 0.0
                    self.segment_idx = 0
        except Exception as e:
            self.get_logger().error(f"Falha ao carregar novos waypoints: {e}")

    def traj_callback(self) -> None:
        desire_pos = np.zeros(3)
        desire_vel = np.zeros(3)
        desire_yaw = 0.0

        # --- State 0: Aguardando no Chão ---
        if self.mission_state == "IDLE":
            # Mantém estático onde pousou
            desire_pos = self.idle_pos.copy()
            desire_vel = np.zeros(3)

        # --- State 1: Decolagem e Interceptação do WP inicial ---
        elif self.mission_state == "TAKEOFF":
            self.recovery_time += self.dt
            desire_pos = self.recovery_traj.sample_position(self.recovery_time)
            desire_vel = self.recovery_traj.sample_velocity(self.recovery_time)
            
            # Ao alcançar o ponto inicial, troca o estado para TRACKING
            if self.recovery_time >= self.recovery_duration:
                self.get_logger().info('TAKEOFF finalizado. Executando a rota matemática...')
                self.traj_time = 0.0
                self.segment_idx = 0
                self.mission_state = "TRACKING"

        # --- State 2: Rota Normal ---
        elif self.mission_state == "TRACKING":
            self.traj_time += self.dt
            desire_pos, desire_vel, desire_yaw, _ = self.path_manager.get_desired_state(self.traj_time)

            local_time = self.traj_time - self.path_manager.start_times[self.segment_idx]
            if local_time >= self.path_manager.segment_times[self.segment_idx]:
                self.get_logger().info(f'Stabilizing/Approaching waypoint {self.segment_idx + 1}')
                self.mission_state = "STABILIZING"
        
        # --- State 3: Estabilização de Waypoint ---
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
                    self.get_logger().info('Fim da rota atingido. Iniciando POUSO (LANDING)...')
                    ground_pos = self.curr_position.copy()
                    ground_pos[2] = 0.0 # Define a altitude zero exatamente abaixo de onde parou
                    
                    dist = np.linalg.norm(self.curr_position - ground_pos)
                    self.recovery_duration = max(3.0, dist / 2.0) 
                    self.recovery_traj = PolinomialTraj(self.curr_position, ground_pos, self.recovery_duration)
                    self.recovery_traj.generate_path()
                    
                    self.recovery_time = 0.0
                    self.mission_state = "LANDING"

        # --- State 4: Pouso Automático ---
        elif self.mission_state == "LANDING":
            self.recovery_time += self.dt
            desire_pos = self.recovery_traj.sample_position(self.recovery_time)
            desire_vel = self.recovery_traj.sample_velocity(self.recovery_time)
            
            if self.recovery_time >= self.recovery_duration:
                self.get_logger().info('Pouso finalizado com sucesso. Retornando ao estado IDLE.')
                self.idle_pos = self.curr_position.copy()
                self.idle_pos[2] = 0.0 # Garante fixação no chão
                self.mission_state = "IDLE"

        # --- State 5: Piloto Manual ---
        elif self.mission_state == "MANUAL_MODE":
            desire_pos = self.curr_position.copy()
            desire_vel = np.zeros(3)
            desire_yaw = self.paused_yaw

        # --- State 6: Transição Suave Manual -> Auto ---
        elif self.mission_state == "RECOVERING":
            self.recovery_time += self.dt
            desire_pos = self.recovery_traj.sample_position(self.recovery_time)
            desire_vel = self.recovery_traj.sample_velocity(self.recovery_time)
            desire_yaw = self.paused_yaw

            if self.recovery_time >= self.recovery_duration:
                self.get_logger().info('UAV successfully repositioned on route. Resuming mission.')
                if self.traj_time >= self.path_manager.total_time:
                    self.mission_state = "STABILIZING"
                else:
                    self.mission_state = "TRACKING"

        # --- Publicação de Referência de Odometria ---
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