"""Run the GUI's ROS 2 bridge in its own process, isolated from Qt.

``rclpy`` (FastDDS) and ``PySide6.QtWebEngineCore`` (Chromium) each link their
own native C++ allocators and libraries. Loading both in the same process
causes C++ heap corruption (``free(): invalid pointer``) during garbage
collection. This module keeps all ``rclpy`` imports inside ``run_bridge`` so
the parent Qt process never loads ``rclpy`` and the spawned bridge process
never loads ``PySide6``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any

from scipy.spatial.transform import Rotation

# Throttle telemetry sent to the GUI to 30 Hz; the physics stream runs at 1000 Hz.
EMIT_PERIOD: float = 1.0 / 30.0


def run_bridge(to_gui: mp.Queue, from_gui: mp.Queue, ros_args: list[str]) -> None:
    """Spin the ROS 2 bridge node in an isolated process until ``quit`` is received."""
    for candidate in ('/workspace/src/m300_sim/scripts', '/workspace/src/m300_sim'):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    from geometry_msgs.msg import AccelStamped, WrenchStamped
    from nav_msgs.msg import Odometry
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import Bool, Empty

    class GuiBridgeNode(Node):
        """Forward telemetry to the GUI and publish GUI commands to ROS 2."""

        def __init__(self) -> None:
            super().__init__('gui_bridge_node')
            self.last_emit_time = 0.0
            self.last_control = {'fz': 0.0, 'tx': 0.0, 'ty': 0.0, 'tz': 0.0}
            self.last_accel = {
                'ax': 0.0,
                'ay': 0.0,
                'az': 0.0,
                'alpha_p': 0.0,
                'alpha_q': 0.0,
                'alpha_r': 0.0,
            }

            self.create_subscription(
                Odometry, '/m300_sim/telemetry_topic', self.telemetry_callback, 1
            )
            self.create_subscription(
                WrenchStamped, '/m300_sim/control_topic', self.ctrl_callback, 1
            )
            self.create_subscription(
                AccelStamped, '/m300_sim/acceleration_topic', self.accel_callback, 1
            )
            self.create_subscription(
                Bool, '/m300_sim/nn_enable', self.nn_enable_callback, 10
            )
            self.pub_start = self.create_publisher(Empty, '/m300_sim/start_mission', 10)
            self.pub_nn_enable = self.create_publisher(Bool, '/m300_sim/nn_enable', 10)

            self.get_logger().info('GUI Bridge Node started. GCS Dashboard Online.')

        def _send(self, item: tuple[str, Any]) -> None:
            try:
                to_gui.put_nowait(item)
            except queue.Full:
                pass

        def handle_command(self, cmd: str, value: Any) -> None:
            """Execute a command received from the GUI process."""
            if cmd == 'start':
                self.get_logger().info(
                    'RUN button pressed. Sending takeoff authorization...'
                )
                self.pub_start.publish(Empty())
            elif cmd == 'nn':
                msg = Bool()
                msg.data = bool(value)
                self.pub_nn_enable.publish(msg)
                status = 'ENABLED' if msg.data else 'DISABLED'
                self.get_logger().info(
                    f'Neural Network residual compensation {status} via GUI.'
                )

        def nn_enable_callback(self, msg: Bool) -> None:
            """Forward NN toggle events (GUI or Xbox controller) to the GUI."""
            self._send(('nn', bool(msg.data)))

        def telemetry_callback(self, msg: Odometry) -> None:
            """Convert and throttle odometry into a plain dict for the GUI."""
            now = time.monotonic()
            if now - self.last_emit_time < EMIT_PERIOD:
                return
            self.last_emit_time = now

            ori = msg.pose.pose.orientation
            roll, pitch, yaw = Rotation.from_quat(
                [ori.x, ori.y, ori.z, ori.w]
            ).as_euler('xyz')
            pos = msg.pose.pose.position
            lin = msg.twist.twist.linear
            ang = msg.twist.twist.angular

            data: dict[str, float] = {
                't': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'x': pos.x,
                'y': pos.y,
                'z': pos.z,
                'roll': float(roll),
                'pitch': float(pitch),
                'yaw': float(yaw),
                'u': lin.x,
                'v': lin.y,
                'w': lin.z,
                'p': ang.x,
                'q': ang.y,
                'r': ang.z,
                **self.last_accel,
                **self.last_control,
            }
            self._send(('telemetry', data))

        def ctrl_callback(self, msg: WrenchStamped) -> None:
            """Cache the latest control wrench [N], [N·m] (body frame)."""
            self.last_control['fz'] = msg.wrench.force.z
            self.last_control['tx'] = msg.wrench.torque.x
            self.last_control['ty'] = msg.wrench.torque.y
            self.last_control['tz'] = msg.wrench.torque.z

        def accel_callback(self, msg: AccelStamped) -> None:
            """Cache the latest accelerations [m/s²], [rad/s²] (body frame)."""
            self.last_accel['ax'] = msg.accel.linear.x
            self.last_accel['ay'] = msg.accel.linear.y
            self.last_accel['az'] = msg.accel.linear.z
            self.last_accel['alpha_p'] = msg.accel.angular.x
            self.last_accel['alpha_q'] = msg.accel.angular.y
            self.last_accel['alpha_r'] = msg.accel.angular.z

    rclpy.init(args=ros_args)
    node = GuiBridgeNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            while True:
                try:
                    cmd, value = from_gui.get_nowait()
                except queue.Empty:
                    break
                if cmd == 'quit':
                    return
                node.handle_command(cmd, value)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
