import os
import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('m300_sim')


    aircraft_config_path = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')

    aircraft_config_arg = DeclareLaunchArgument(
        'aircraft_config',
        default_value=aircraft_config_path,
        description='Caminho para o YAML com parametros fisicos do drone.'
    )

    aircraft_config = LaunchConfiguration('aircraft_config')

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    bag_path = os.path.join('src', 'm300_sim', 'bags', f'voo_online_{timestamp}')

    record_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', 
             '/m300_sim/telemetry_topic', 
             '-o', bag_path],
        output='screen'
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node"
    )

    joy_map_node = Node(
        package='m300_sim',
        executable='joy_sub',
        name='joy_mapper',
    )

    ctrl_node = Node(
        package='m300_sim',
        executable='ctrl_sub',
        name='controller_node',
        output='screen',
        parameters=[aircraft_config] # <--- Lendo o YAML da Aeronave!
    )

    dyn_node = Node(
        package='m300_sim',
        executable='dyn_pub',
        name='quadcopter_node',
        output='screen',
        parameters=[aircraft_config] # <--- Lendo o YAML da Aeronave!
    )

    ld = LaunchDescription()
    ld.add_action(aircraft_config_arg)
    ld.add_action(record_bag)
    ld.add_action(joy_node)
    ld.add_action(joy_map_node)
    ld.add_action(ctrl_node)
    ld.add_action(dyn_node)

    return ld