import os
import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('m300_sim')

    # 1. Caminhos para os arquivos YAML (Carregando tanto da aeronave quanto da missão)
    aircraft_config_path = os.path.join(pkg_share, 'config', 'aircraft_params.yaml')
    mission_config_path = os.path.join(pkg_share, 'config', 'mission_params.yaml')

    # 2. Argumentos de Lançamento
    aircraft_config_arg = DeclareLaunchArgument(
        'aircraft_config',
        default_value=aircraft_config_path,
        description='Caminho para o YAML com parametros fisicos do drone.'
    )
    mission_config_arg = DeclareLaunchArgument(
        'mission_config',
        default_value=mission_config_path,
        description='Caminho para o YAML com parametros da rota (waypoints).'
    )

    aircraft_config = LaunchConfiguration('aircraft_config')
    mission_config = LaunchConfiguration('mission_config')

    # 3. Gravação do ROS Bag (Atualizado para gravar todos os tópicos importantes da transição)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    bag_path = os.path.join('src', 'm300_sim', 'bags', f'voo_unificado_{timestamp}')

    record_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', 
             '/m300_sim/telemetry_topic', 
             '/m300_sim/trajectory_topic',
             '/m300_sim/flight_mode',
             '/m300_sim/manual_cmd',
             '-o', bag_path],
        output='screen'
    )

    # 4. Definição de TODOS os Nós

    # a) Nós de Joystick e Mapeamento
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

    # b) Nó de Trajetória (Agora com a Máquina de Estados Inteligente)
    traj_node = Node(
        package='m300_sim',
        executable='traj_pub',
        name='trajectory_node',
        output='screen',
        parameters=[mission_config] # <--- Lendo o YAML da Missão!
    )

    # c) Nós de Controle e Dinâmica
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

    # 5. Adicionando as ações na Launch Description
    ld = LaunchDescription()
    
    ld.add_action(aircraft_config_arg)
    ld.add_action(mission_config_arg)
    
    ld.add_action(record_bag)
    
    ld.add_action(joy_node)
    ld.add_action(joy_map_node)
    ld.add_action(traj_node)
    ld.add_action(ctrl_node)
    ld.add_action(dyn_node)

    return ld