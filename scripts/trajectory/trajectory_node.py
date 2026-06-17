import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
import numpy as np

from nav_msgs.msg import Odometry

from .trajectory import PathManager

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')

        # Iniciando trajetória
        self.declare_parameter('waypoints', [0.0, 0.0, 0.0, 0.0, 0.0, -10.0])
        self.declare_parameter('yaw_mode', 'forward')

        wp_flat = self.get_parameter('waypoints').value
        yaw_mode_param = self.get_parameter('yaw_mode').value

        waypoints = [wp_flat[i:i+3] for i in range(0, len(wp_flat), 3)]
        self.path_manager = PathManager(waypoints, yaw_mode=yaw_mode_param)
        self.path_manager.generate_path()

        # FSM 
        self.mission_state = "TRACKING"
        self.traj_time = 0.0
        self.segment_idx = 0
        self.curr_position = np.zeros(3)
        self.wp_tolerance = 0.5
        self.dt = 1/100

        self.traj_pub = self.create_publisher(Odometry, '/m300_sim/trajectory_topic', 10)
        
        self.state_sub = self.create_subscription(Odometry, '/m300_sim/telemetry_topic', self.pose_callback, 10)

        self.create_timer(self.dt, self.traj_callback)

        self.get_logger().info(f'Traj publisher iniciado.')

    def pose_callback(self, msg:Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        self.curr_position[0:3] = px, py, pz 

    def traj_callback(self):
        desire_pos, desire_vel, desire_yaw, _ = self.path_manager.get_desired_state(self.traj_time)

        if self.mission_state == "TRACKING":
            self.traj_time += self.dt

            local_time = self.traj_time - self.path_manager.start_times[self.segment_idx]
            if local_time >= self.path_manager.segment_times[self.segment_idx]:
                self.get_logger().info(f'Estabilizando no waypoint {self.segment_idx + 1}')
                self.mission_state = "STABILIZING"
        
        elif self.mission_state == "STABILIZING":
            target_pos = self.path_manager.waypoints[self.segment_idx + 1][0:3]
            error = np.linalg.norm(self.curr_position - target_pos)

            if error < self.wp_tolerance and self.segment_idx < (len(self.path_manager.segments) - 1):
                self.segment_idx += 1   
                self.get_logger().info(f'Prosseguindo para waypoint {self.segment_idx + 2}')
                self.mission_state = "TRACKING"

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Posições
        msg.pose.pose.position.x = float(desire_pos[0])
        msg.pose.pose.position.y = float(desire_pos[1])
        msg.pose.pose.position.z = float(desire_pos[2])

        # Quaterinon
        rot = Rotation.from_euler('xyz', [0.0, 0.0, desire_yaw])
        qx, qy, qz, qw = rot.as_quat()    
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Velocidades
        msg.twist.twist.linear.x = float(desire_vel[0])
        msg.twist.twist.linear.y = float(desire_vel[1])
        msg.twist.twist.linear.z = float(desire_vel[2])

        self.traj_pub.publish(msg)


def main(args=None):
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



