import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import numpy as np
from scipy.spatial.transform import Rotation
import threading

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry

from .control import CascadeController

class TickController(Node):
    def __init__(self):
        super().__init__("controller_node")
        # Declarar os parâmetros necessários para o controle
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

        # Criar um objeto Dummy temporário para não quebrar a assinatura da classe CascadeController
        class DummyModel: pass
        ac_model = DummyModel()
        ac_model.mass = self.get_parameter('mass').value
        ac_model.max_ascent_speed = self.get_parameter('max_ascent_speed').value
        ac_model.max_descent_speed = self.get_parameter('max_descent_speed').value
        ac_model.cruise_speed = self.get_parameter('cruise_speed').value
        ac_model.max_tilt_angle = self.get_parameter('max_tilt_angle').value
        ac_model.max_roll_pitch_rate = self.get_parameter('max_roll_pitch_rate').value
        ac_model.max_yaw_rate = self.get_parameter('max_yaw_rate').value

        # Instanciar o controlador com os dados do YAML
        self.controller = CascadeController(ac_model)

        self.ctrl_pub = self.create_publisher(Float64MultiArray, '/drone_simulator/control_topic', 10)

        self.traj_sub = self.create_subscription(PoseStamped, '/drone_simulator/trajectory_topic', 
                                            self.trajectory_callback, 10)
        
        self.dyn_sub = self.create_subscription(Odometry, '/drone_simulator/telemetry_topic', 
                                                self.state_callback, 10)


        self.get_logger().info(f'Initializing controller publisher')

        self.pos_x, self.pos_y, self.pos_z = 0.0, 0.0, 0.0
        self.yaw = 0.0

        self.state = np.zeros(12)

        self.tick = 0


    def state_callback(self, msg:Odometry):
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

        self.controller.update_state(self.state)

        u_total, tau = self.controller.control_px4(self.tick)

        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = [float(u_total), float(tau[0]), float(tau[1]), float(tau[2])]

        self.ctrl_pub.publish(ctrl_msg)


        self.tick += 1

    def trajectory_callback(self, msg:PoseStamped):
        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        self.pos_z = msg.pose.position.z

        quat_x = msg.pose.orientation.x
        quat_y = msg.pose.orientation.y
        quat_z = msg.pose.orientation.z
        quat_w = msg.pose.orientation.w
        
        rot = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w])

        roll, pitch, self.yaw = rot.as_euler('xyz')

        waypoint = np.array([self.pos_x, self.pos_y, self.pos_z, self.yaw])
        self.controller.desired_state(waypoint)


class Controller(Node):
    def __init__(self):
        super().__init__("controller_node")
        # Declarar os parâmetros necessários para o controle
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

        # Criar um objeto Dummy temporário para não quebrar a assinatura da classe CascadeController
        class DummyModel: pass
        ac_model = DummyModel()
        ac_model.mass = self.get_parameter('mass').value
        ac_model.max_ascent_speed = self.get_parameter('max_ascent_speed').value
        ac_model.max_descent_speed = self.get_parameter('max_descent_speed').value
        ac_model.cruise_speed = self.get_parameter('cruise_speed').value
        ac_model.max_tilt_angle = self.get_parameter('max_tilt_angle').value
        ac_model.max_roll_pitch_rate = self.get_parameter('max_roll_pitch_rate').value
        ac_model.max_yaw_rate = self.get_parameter('max_yaw_rate').value

        # Instanciar o controlador com os dados do YAML
        self.controller = CascadeController(ac_model)

        self.pos_x, self.pos_y, self.pos_z = 0.0, 0.0, 0.0
        self.yaw = 0.0
        self.state = np.zeros(12)
        self.u_total = 0.0
        self.tau = np.zeros(3)

        self.lock = threading.Lock()

        # Callback groups
        self.cb_groups_subs = MutuallyExclusiveCallbackGroup()
        self.cb_group_50hz = MutuallyExclusiveCallbackGroup()
        self.cb_group_250hz = MutuallyExclusiveCallbackGroup()
        self.cb_group_1000hz = MutuallyExclusiveCallbackGroup()

        # ==== Subscribers ==== #
        # Trajectory
        self.traj_sub = self.create_subscription(Odometry, '/drone_simulator/trajectory_topic', 
                                            self.trajectory_callback, 10, callback_group=self.cb_groups_subs)
        # Dynamic
        self.dyn_sub = self.create_subscription(Odometry, '/drone_simulator/telemetry_topic', 
                                                self.state_callback, 10, callback_group=self.cb_groups_subs)
        
        # ==== Publisher ==== #
        self.ctrl_pub = self.create_publisher(Float64MultiArray, '/drone_simulator/control_topic', 10)


        # ==== Timers ==== #
        self.create_timer(1/50, self.loop_50hz, callback_group=self.cb_group_50hz)
        self.create_timer(1/250, self.loop_250hz, callback_group=self.cb_group_250hz)
        self.create_timer(1/1000, self.loop_1000hz, callback_group=self.cb_group_1000hz)


    def loop_50hz(self):
        with self.lock:
            self.controller._xy_pos_control()
            self.controller._z_pos_control()
            
            self.controller._xy_vel_control(dt=1/50)
            self.controller._z_vel_control(dt=1/50)

            self.u_total = self.controller._acel_to_atitude()    

    def loop_250hz(self):
        with self.lock:
            self.controller._angle_control()
        
    def loop_1000hz(self):
        with self.lock:
            self.tau = self.controller._angular_rate_control(dt=1/1000)

            ctrl_msg = Float64MultiArray()
            ctrl_msg.data = [float(self.u_total), float(self.tau[0]), float(self.tau[1]), float(self.tau[2])]

        self.ctrl_pub.publish(ctrl_msg)
        
    def state_callback(self, msg:Odometry):
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
       

    def trajectory_callback(self, msg:Odometry):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z

        quat_x = msg.pose.pose.orientation.x
        quat_y = msg.pose.pose.orientation.y
        quat_z = msg.pose.pose.orientation.z
        quat_w = msg.pose.pose.orientation.w
        
        rot = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w])

        _, _, self.yaw = rot.as_euler('xyz')

        vel_x = msg.twist.twist.linear.x    
        vel_y = msg.twist.twist.linear.y    
        vel_z = msg.twist.twist.linear.z  

        vel = np.array([vel_x, vel_y, vel_z])  
        waypoint = np.array([self.pos_x, self.pos_y, self.pos_z, self.yaw])

        with self.lock:
            self.controller.desired_state(waypoint, vel)


        
    



def main(args=None):
    rclpy.init(args=args)
    ctrl_node = Controller()

    executor = MultiThreadedExecutor()
    executor.add_node(ctrl_node)
    try:
        executor.spin()
        # rclpy.spin(ctrl_node)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl_node.destroy_node()
        rclpy.shutdown()
    
if __name__ == "__main__":
    main()