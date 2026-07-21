import sys
import os
import time
import threading

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from scipy.spatial.transform import Rotation

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt, QObject, Signal
from gui.widgets.main_window import MainWindow

# ==============================================================================
# SINAIS DA PONTE
# ==============================================================================
class ROS2Signals(QObject):
    telemetry_updated = Signal(dict)

# ==============================================================================
# NÓ INVISÍVEL DO ROS 2
# ==============================================================================
class GuiBridgeNode(Node):
    def __init__(self, signals: ROS2Signals):
        super().__init__('gui_bridge_node')
        self.signals = signals
        self.last_emit_time = 0.0
        self.emit_rate = 1.0 / 30.0  # 30 FPS para não travar a GUI
        
        # Assina a telemetria do drone
        self.sub_telemetry = self.create_subscription(Odometry, '/m300_sim/telemetry_topic', self.telemetry_callback, 10)
        
        # Cria o publicador do Gatilho
        self.pub_start = self.create_publisher(Empty, '/m300_sim/start_mission', 10)
        
        self.get_logger().info("GUI Bridge Node iniciado. Painel GCS Online.")

    def trigger_mission(self):
        """Publica o sinal que acorda a Máquina de Estados do Drone."""
        self.get_logger().info("Botão RUN pressionado. Enviando autorização de decolagem...")
        self.pub_start.publish(Empty())

    def telemetry_callback(self, msg: Odometry):
        current_time = time.time()
        if (current_time - self.last_emit_time) < self.emit_rate:
            return 
        self.last_emit_time = current_time

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        rot = Rotation.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler('xyz')

        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z
        p = msg.twist.twist.angular.x
        q = msg.twist.twist.angular.y
        r = msg.twist.twist.angular.z

        data = {
            't': msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9),
            'x': x, 'y': y, 'z': z,
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'u': u, 'v': v, 'w': w,
            'p': p, 'q': q, 'r': r
        }
        self.signals.telemetry_updated.emit(data)

def spin_ros_node(node):
    rclpy.spin(node)
    node.destroy_node()

def apply_global_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.Base, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

def main(args=None):
    # 1. A REGRA DE OURO DO PYQT/PYSIDE: Instanciar a Aplicação PRIMEIRO!
    app = QApplication(sys.argv)
    apply_global_dark_theme(app)

    # 2. Só então inicializamos o ROS 2 e os Sinais!
    rclpy.init(args=args)
    ros_signals = ROS2Signals()
    gui_node = GuiBridgeNode(ros_signals)
    
    ros_thread = threading.Thread(target=spin_ros_node, args=(gui_node,), daemon=True)
    ros_thread.start()

    # 3. Cria e conecta a Janela
    window = MainWindow()
    ros_signals.telemetry_updated.connect(window.tab_outputs.receive_online_data)
    window.tab_inputs.btn_run_sim.clicked.connect(gui_node.trigger_mission)
    
    window.show()
    exit_code = app.exec()
    
    rclpy.shutdown()
    ros_thread.join(timeout=1.0)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()