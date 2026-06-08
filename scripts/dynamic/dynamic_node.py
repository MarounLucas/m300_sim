import rclpy
from rclpy.node import Node

import numpy as np
import math
from scipy.spatial.transform import Rotation

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

from .mixer import Mixer
from .quadcopter_model import six_dof_model
from .numerical_integration import forward_euler
from .wind_manager import WindManager

D2R = math.pi / 180
R2D = D2R ** (-1)


class DynamicNode(Node):
    def __init__(self):
        super().__init__("quadcopter_node")
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('mass', 6.47), ('jx', 0.214), ('jy', 0.200), ('jz', 0.389), ('jxz', 0.0),
                ('cd', 0.25), ('k_t0', 0.0002433), ('k_q0', 0.00000692), ('tau', 0.015),
                ('kp', 1.0), ('xi', 1.0), ('dx_arm', 0.335), ('dy_fw', 0.365), ('dy_bw', 0.315),
                ('dy_bw', 0.315), ('omega_min', 60.0), ('omega_max', 500.0)
            ]
        )

        # Resgatar os valores carregados pelo YAML
        mass = self.get_parameter('mass').value
        jx = self.get_parameter('jx').value
        jy = self.get_parameter('jy').value
        jz = self.get_parameter('jz').value
        jxz = self.get_parameter('jxz').value
        cd = self.get_parameter('cd').value
        k_t0 = self.get_parameter('k_t0').value
        k_q0 = self.get_parameter('k_q0').value
        tau = self.get_parameter('tau').value
        kp = self.get_parameter('kp').value
        xi = self.get_parameter('xi').value
        dx_arm = self.get_parameter('dx_arm').value
        dy_fw = self.get_parameter('dy_fw').value
        dy_bw = self.get_parameter('dy_bw').value
        omega_min = self.get_parameter('omega_min').value
        omega_max = self.get_parameter('omega_max').value
        


        self.ac_params = np.array([
            mass, jx, jy, jz, jxz, 
            cd, k_t0, k_q0,
            tau, kp, xi,
            dx_arm, dy_fw, dy_bw, 
        ], dtype=np.float64)

        class DummyModel: pass
        self.ac_model = DummyModel()
        self.ac_model.mass = mass
        self.ac_model.k_t0 = k_t0
        self.ac_model.k_q0 = k_q0
        self.ac_model.dx_arm = dx_arm
        self.ac_model.dy_fw = dy_fw
        self.ac_model.dy_bw = dy_bw
        self.ac_model.omega_min = omega_min
        self.ac_model.omega_max = omega_max

        self.mixer = Mixer(self.ac_model)

        # Init Mixer
        self.mixer = Mixer(self.ac_model)
        
        # X, Y, Z, Yaw
        self.target_state = np.zeros(4) 

        # Initializing engines
        self.u_virtual = np.zeros(4)
        omega_hover = np.sqrt((self.ac_model.mass * 9.81) / (4.0 * self.ac_model.k_t0))

        # Initial state 
        self.init_state = np.array([
            0.0, 0.0, 0.0,                                    # u, v, w
            0.0, 0.0, 0.0,                                    # p, q, r
            0.0 * D2R, 0.0 * D2R, 0.0 * D2R,                  # phi, theta, psi
            0.0, 0.0, 0.0,                                    # px, py, pz
            omega_hover, 0.0,                                 # w_m1, alpha_m1
            omega_hover, 0.0,                                 # w_m2, alpha_m2
            omega_hover, 0.0,                                 # w_m3, alpha_m3
            omega_hover, 0.0                                  # w_m4, alpha_m4
        ])
        self.state = self.init_state.copy()

        # Wind
        self.curr_wind = np.zeros(3)
        w_type = "constant"
        w_mag, w_head, w_elev, w_gust = 5.0, 45.0, 45.0, 0.0

        self.wind_manager = WindManager(
            wind_type=w_type, 
            magnitude=w_mag, 
            heading=w_head * D2R, 
            elevation=w_elev * D2R, 
            gust_magnitude=w_gust
        )

        self.dt = 1/1000
        self.tick_count = 0

        self.dyn_pub = self.create_publisher(Odometry,'/drone_simulator/telemetry_topic', 10)
        self.get_logger().info(f'Telemetry Publisher Iniciado')

        self.ctrl_sub = self.create_subscription(Float64MultiArray, '/drone_simulator/control_topic', self.cmd_callback, 10)
        self.create_timer(self.dt, self.physics_loop)


    def cmd_callback(self, msg:Float64MultiArray):
        self.u_virtual = np.array(msg.data)
        

    def physics_loop(self):
        # Vento a 100hz
        if self.tick_count % 10 == 0:
            t_sim = self.tick_count * self.dt
            current_pos = self.state[9:12]
            self.curr_wind = self.wind_manager.get_wind(t_sim, current_pos)

        w_cmds = self.mixer.compute_motor_speed(self.u_virtual)

        dx = six_dof_model(self.state, self.ac_params, self.curr_wind, w_cmds)

        self.state = forward_euler(self.state, dx, self.dt)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        # Posição
        msg.pose.pose.position.x = float(self.state[9])
        msg.pose.pose.position.y = float(self.state[10])
        msg.pose.pose.position.z = float(self.state[11])

        # Orientação
        rot = Rotation.from_euler("xyz", [self.state[6], self.state[7], self.state[8]])
        qx, qy, qz, qw  = rot.as_quat()
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Velocidade Linear 
        msg.twist.twist.linear.x = self.state[0]
        msg.twist.twist.linear.y = self.state[1]
        msg.twist.twist.linear.z = self.state[2]

        # Velocidade angular 
        msg.twist.twist.angular.x = self.state[3]
        msg.twist.twist.angular.y = self.state[4]
        msg.twist.twist.angular.z = self.state[5]

        self.dyn_pub.publish(msg)


        if self.tick_count % 100 == 0:
            # Extraindo ângulos de Euler para visualização mais humana
            rot = Rotation.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            roll, pitch, yaw = rot.as_euler('xyz', degrees=True) # Em graus para o log!
            
            self.get_logger().info(
                f'--- M350 Telemetria ---\n'
                f'Pos (m): X={msg.pose.pose.position.x:.2f}, Y={msg.pose.pose.position.y:.2f}, Z={msg.pose.pose.position.z:.2f}\n'
                f'Vel (m/s): u={msg.twist.twist.linear.x:.2f}, v={msg.twist.twist.linear.y:.2f}, w={msg.twist.twist.linear.z:.2f}\n'
                f'Atitude (deg): Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}\n'
            )
        
        self.tick_count +=1

def main(args=None):
    rclpy.init(args=args)
    node = DynamicNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    
    