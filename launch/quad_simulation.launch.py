"""ROS 2 launch description for the M350 quadcopter simulation stack."""

from __future__ import annotations

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _sync_source_tree() -> None:
    """Synchronize source scripts into the ROS 2 install tree before spawning nodes."""
    src_scripts = '/workspace/src/m300_sim/scripts'
    if os.path.isdir(src_scripts):
        if src_scripts not in sys.path:
            sys.path.insert(0, src_scripts)
        try:
            import paths

            paths.sync_ros_install_scripts()
        except Exception:
            pass


def generate_launch_description() -> LaunchDescription:
    """Build the ROS 2 launch description for the simulator nodes and GUI."""
    _sync_source_tree()
    pkg_share = get_package_share_directory('m300_sim')

    aircraft_config_path = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')
    mission_config_path = os.path.join(pkg_share, 'config', 'mission_params.yaml')

    aircraft_config_arg = DeclareLaunchArgument(
        'aircraft_config', default_value=aircraft_config_path
    )
    mission_config_arg = DeclareLaunchArgument(
        'mission_config', default_value=mission_config_path
    )

    aircraft_config = LaunchConfiguration('aircraft_config')
    mission_config = LaunchConfiguration('mission_config')

    joy_node = Node(package='joy', executable='joy_node', name='joy_node')
    joy_map_node = Node(
        package='m300_sim', executable='joy_sub', name='joystick_node'
    )
    traj_node = Node(
        package='m300_sim',
        executable='traj_pub',
        name='trajectory_node',
        parameters=[mission_config],
    )
    ctrl_node = Node(
        package='m300_sim',
        executable='ctrl_sub',
        name='controller_node',
        parameters=[aircraft_config],
    )
    dyn_node = Node(
        package='m300_sim',
        executable='dyn_pub',
        name='quadcopter_node',
        parameters=[aircraft_config],
    )

    gui_node = Node(
        package='m300_sim',
        executable='gui_node',
        name='simulador_gui_node',
        output='screen',
        emulate_tty=True,
        additional_env={
            'QTWEBENGINE_CHROMIUM_FLAGS': (
                '--no-sandbox --disable-dev-shm-usage --ignore-gpu-blocklist '
                '--enable-webgl --enable-gpu-rasterization'
            ),
        },
    )

    # WebSocket server streaming telemetry to the embedded Unity WebGL view
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{'port': 9090}],
    )

    # Shut down all simulation nodes automatically when the GUI window closes
    sys_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=gui_node,
            on_exit=[EmitEvent(event=Shutdown(reason='GUI Window Closed'))],
        )
    )

    ld = LaunchDescription()
    ld.add_action(aircraft_config_arg)
    ld.add_action(mission_config_arg)
    ld.add_action(joy_node)
    ld.add_action(joy_map_node)
    ld.add_action(traj_node)
    ld.add_action(ctrl_node)
    ld.add_action(dyn_node)
    ld.add_action(rosbridge_node)
    ld.add_action(gui_node)
    ld.add_action(sys_shutdown)

    return ld
