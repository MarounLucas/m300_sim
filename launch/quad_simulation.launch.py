import os
import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, EmitEvent, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('m300_sim')

    aircraft_config_path = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')
    mission_config_path = os.path.join(pkg_share, 'config', 'mission_params.yaml')

    aircraft_config_arg = DeclareLaunchArgument('aircraft_config', default_value=aircraft_config_path)
    mission_config_arg = DeclareLaunchArgument('mission_config', default_value=mission_config_path)

    aircraft_config = LaunchConfiguration('aircraft_config')
    mission_config = LaunchConfiguration('mission_config')

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    bag_path = os.path.join('src', 'm300_sim', 'bags', f'voo_unificado_{timestamp}')

    record_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '/m300_sim/telemetry_topic', '/m300_sim/trajectory_topic',
             '/m300_sim/flight_mode', '/m300_sim/manual_cmd', '-o', bag_path],
        output='screen'
    )

    joy_node = Node(package="joy", executable="joy_node", name="joy_node")
    joy_map_node = Node(package='m300_sim', executable='joy_sub', name='joystick_node')
    traj_node = Node(package='m300_sim', executable='traj_pub', name='trajectory_node', parameters=[mission_config])
    ctrl_node = Node(package='m300_sim', executable='ctrl_sub', name='controller_node', parameters=[aircraft_config])
    dyn_node = Node(package='m300_sim', executable='dyn_pub', name='quadcopter_node', parameters=[aircraft_config])
    
    gui_node = Node(
        package='m300_sim',
        executable='gui_node',
        name='simulador_gui_node',
        output='screen'
    )

    # NOVO: O Servidor WebSocket que vai transmitir dados para a Unity (WebGL)
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{'port': 9090}]
    )

    # A MÁGICA DE DESLIGAMENTO: Se a GUI fechar, mata tudo automaticamente.
    sys_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=gui_node,
            on_exit=[EmitEvent(event=Shutdown(reason='GUI Window Closed'))]
        )
    )

    ld = LaunchDescription()
    ld.add_action(aircraft_config_arg)
    ld.add_action(mission_config_arg)
    ld.add_action(record_bag)
    ld.add_action(joy_node)
    ld.add_action(joy_map_node)
    ld.add_action(traj_node)
    ld.add_action(ctrl_node)
    ld.add_action(dyn_node)
    ld.add_action(rosbridge_node) # <- Adicionado no Launch
    ld.add_action(gui_node)
    ld.add_action(sys_shutdown)

    return ld