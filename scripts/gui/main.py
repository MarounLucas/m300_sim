"""ROS 2 and PySide6 graphical interface bridge node.

This module sets up a Qt GUI application that runs alongside a ROS 2 node in
a separate thread. It manages a dark theme application, throttles incoming UAV
telemetry to maintain GUI performance, and provides trigger signals to the UAV.
"""

import os
import sys
import threading
import time
from typing import Dict, List, Optional

# Inject project root into Python path to allow local imports.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from geometry_msgs.msg import WrenchStamped, AccelStamped
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Local imports
from gui.widgets.main_window import MainWindow


os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--ignore-gpu-blocklist --enable-webgl --enable-gpu-rasterization"
os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = "9999"



# ==============================================================================
# BRIDGE SIGNALS
# ==============================================================================
class ROS2Signals(QObject):
    """Defines custom Qt signals for ROS 2 bridge communication.

    Attributes:
        telemetry_updated (Signal): Emits a dictionary with updated telemetry.
    """
    telemetry_updated = Signal(dict)


# ==============================================================================
# INVISIBLE ROS 2 NODE
# ==============================================================================
class GuiBridgeNode(Node):
    """Invisible ROS 2 node bridging telemetry data to the PySide6 GUI.

    Subscribes to UAV telemetry and publishes mission triggers, while
    throttling GUI updates to prevent the graphical interface from freezing.

    Attributes:
        signals (ROS2Signals): The Qt signal object for UI communication.
        last_emit_time (float): Timestamp of the last telemetry emission.
        emit_rate (float): Minimum time interval between telemetry emits.
    """

    def __init__(self, signals: ROS2Signals) -> None:
        """Initializes the GUI bridge node and ROS 2 interfaces.

        Args:
            signals (ROS2Signals): The signal object to emit UI updates.
        """
        super().__init__("gui_bridge_node")
        self.signals = signals
        self.last_emit_time = 0.0
        self.last_control = {'fz': 0.0, 'tx': 0.0, 'ty': 0.0, 'tz': 0.0}
        self.last_accel = {
            'ax': 0.0, 'ay': 0.0, 'az': 0.0, 
            'alpha_p': 0.0, 'alpha_q': 0.0, 'alpha_r': 0.0
}
        
        # 30 FPS throttle to prevent GUI freezing
        self.emit_rate = 1.0 / 30.0  

        # Subscribe to drone telemetry
        self.sub_telemetry = self.create_subscription(
            Odometry, "/m300_sim/telemetry_topic", self.telemetry_callback, 10
        )

        self.ctrl_sub = self.create_subscription(
            WrenchStamped, '/m300_sim/control_topic', self.ctrl_callback, 10
        )

        self.accel_sub = self.create_subscription(
            AccelStamped, '/m300_sim/acceleration_topic', self.accel_callback, 10
        )

        # Create trigger publisher
        self.pub_start = self.create_publisher(
            Empty, "/m300_sim/start_mission", 10
        )

        self.get_logger().info("GUI Bridge Node started. GCS Dashboard Online.")

    def trigger_mission(self) -> None:
        """Publishes the signal to wake the UAV State Machine."""
        self.get_logger().info("RUN button pressed. Sending takeoff authorization...")
        self.pub_start.publish(Empty())

    def telemetry_callback(self, msg: Odometry) -> None:
        """Processes and throttles incoming telemetry data for the GUI.

        Args:
            msg (Odometry): The incoming odometry message.
        """
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
        roll, pitch, yaw = rot.as_euler("xyz")

        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z
        
        p = msg.twist.twist.angular.x
        q = msg.twist.twist.angular.y
        r = msg.twist.twist.angular.z

        data: Dict[str, float] = {
            "t": msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9),
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "u": u, "v": v, "w": w,
            "p": p, "q": q, "r": r,
            "ax": self.last_accel['ax'],
            "ay": self.last_accel['ay'],
            "az": self.last_accel['az'],
            "alpha_p": self.last_accel['alpha_p'],
            "alpha_q": self.last_accel['alpha_q'],
            "alpha_r": self.last_accel['alpha_r'],
            "fz": self.last_control['fz'],
            "tx": self.last_control['tx'],
            "ty": self.last_control['ty'],
            "tz": self.last_control['tz']
        }
        self.signals.telemetry_updated.emit(data)

    def ctrl_callback(self, msg: WrenchStamped):
        self.last_control['fz'] = msg.wrench.force.z
        self.last_control['tx'] = msg.wrench.torque.x
        self.last_control['ty'] = msg.wrench.torque.y
        self.last_control['tz'] = msg.wrench.torque.z

    def accel_callback(self, msg: AccelStamped):
        self.last_accel['ax'] = msg.accel.linear.x
        self.last_accel['ay'] = msg.accel.linear.y
        self.last_accel['az'] = msg.accel.linear.z
        self.last_accel['alpha_p'] = msg.accel.angular.x
        self.last_accel['alpha_q'] = msg.accel.angular.y
        self.last_accel['alpha_r'] = msg.accel.angular.z

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def spin_ros_node(node: Node) -> None:
    """Spins the ROS 2 node in an isolated thread.

    Args:
        node (Node): The ROS 2 node to spin.
    """
    rclpy.spin(node)
    node.destroy_node()


def apply_global_dark_theme(app: QApplication) -> None:
    """Applies a custom Fusion-based dark theme palette to the application.

    Args:
        app (QApplication): The main Qt application instance.
    """
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


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def main(args: Optional[List[str]] = None) -> None:
    """Initializes the GUI and ROS 2 bridge, starting the application loop.

    Args:
        args (Optional[List[str]]): Command-line arguments.
    """
    # 1. The golden rule of PyQt/PySide: Instantiate the Application FIRST!
    app = QApplication(sys.argv)
    apply_global_dark_theme(app)

    # 2. Only then initialize ROS 2 and Signals!
    rclpy.init(args=args)
    ros_signals = ROS2Signals()
    gui_node = GuiBridgeNode(ros_signals)

    ros_thread = threading.Thread(
        target=spin_ros_node, args=(gui_node,), daemon=True
    )
    ros_thread.start()

    # 3. Create and connect the Main Window
    window = MainWindow()
    
    ros_signals.telemetry_updated.connect(
        window.tab_outputs.receive_online_data
    )
    window.tab_inputs.btn_run_sim.clicked.connect(gui_node.trigger_mission)

    window.show()
    exit_code = app.exec()

    rclpy.shutdown()
    ros_thread.join(timeout=1.0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()