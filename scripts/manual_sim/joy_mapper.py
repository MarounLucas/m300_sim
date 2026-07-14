import rclpy
from rclpy.node import Node

import numpy as np  
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from m300_msgs.msg import FlightMode




'''

- AUTO: Simulação "offline" padrão
- MANUAL: 
    1. Armazena o ultimo estado auto;
    2. Mantem hover caso n haja input;
    3. Mapear saída do controle para cmd
    4. cmd - ctrl - dyn 


'''


class JoyMapper(Node):
    def __init__(self):
        super().__init__("joy_mapper")
        
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.fmd_pub = self.create_publisher(FlightMode, '/m300_sim/flight_mode', 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/m300_sim/manual_cmd', 10)

        self.actual_state = np.zeros(4)
        self.postive_yaw = 0
        self.negative_yaw = 0
        
        # Subscriber da dinâmica
        self.dynamic_sub = self.create_subscription(Odometry, "/m300_sim/telemetry_topic", self.dynamic_callback, 10)
        self.flight_mode = FlightMode.AUTO


    def dynamic_callback(self, msg:Odometry):
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y
        z_pos = msg.pose.pose.position.z
        self.actual_state[0:3] = x_pos, y_pos, z_pos

        qx = msg.pose.pose.orientation.x 
        qy = msg.pose.pose.orientation.y 
        qz = msg.pose.pose.orientation.z 
        qw = msg.pose.pose.orientation.w 

        rot = Rotation.from_quat([qx, qy, qz, qw])
        _, _, yaw = rot.as_euler('xyz')
        self.actual_state[3] = yaw


    def apply_deadzone(self, value: float, deadzone: float=0.1):
        if abs(value) < deadzone:
            return 0.0

        sign = 1.0 if value > 0 else -1.0
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def joy_callback(self, msg:Joy):
        flight_mode_ax = msg.axes[7]

        if flight_mode_ax == 1:
            # Apresentar o valor atual da dinâmica
            self.flight_mode = FlightMode.MANUAL
            # self.get_logger().info(f'Estado armazenado (ultimo estado auto): {self.actual_state}')
        elif flight_mode_ax == -1:
            # Apresentar o valor atual da dinâmica
            self.flight_mode = FlightMode.AUTO
            # self.get_logger().info(f'Estado armazenado (ultimo estado manual): {self.actual_state}')
        
        dz = 0.1 

        ax_z_vel = self.apply_deadzone(msg.axes[4], dz)
        ax_z_vel = -ax_z_vel

        ax_north_vel = self.apply_deadzone(msg.axes[1], dz)
        
        ax_east_vel = self.apply_deadzone(msg.axes[0], dz)
        ax_east_vel = -ax_east_vel  # (-1 -> West; 1 -> East)
        
        # Orientação
        # ax_yaw_orientation = msg.axes[3]
        # ax_yaw_orientation = -ax_yaw_orientation  # (-1 anti-horário; 1 horário)


        self.postive_yaw = msg.buttons[5]
        self.negative_yaw = msg.buttons[4]

        ax_yaw_orientation = self.postive_yaw - self.negative_yaw

        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'

        cmd_msg.twist.linear.x = float(ax_north_vel)
        cmd_msg.twist.linear.y = float(ax_east_vel)
        cmd_msg.twist.linear.z = float(ax_z_vel)

        cmd_msg.twist.angular.z = float(ax_yaw_orientation)

        self.cmd_pub.publish(cmd_msg)

        fmd_msg = FlightMode()
        fmd_msg.mode = self.flight_mode
        self.fmd_pub.publish(fmd_msg)



def main(args=None):
    rclpy.init(args=args)
    node = JoyMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    