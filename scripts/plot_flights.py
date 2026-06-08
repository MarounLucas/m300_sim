import os
import glob
import pandas as pd
import numpy as np

# FORÇAR MODO SEM ECRÃ (Para resolver erro do Docker/X11)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

typestore = get_typestore(Stores.ROS2_HUMBLE)

def get_latest_bag(bags_dir: str) -> str:
    bag_folders = glob.glob(os.path.join(bags_dir, 'voo_offline_*'))
    if not bag_folders:
        raise FileNotFoundError(f"Nenhum bag encontrado na pasta: {bags_dir}")
    return max(bag_folders, key=os.path.getmtime)

def process_bag(bag_path: str):
    print(f"==================================================")
    print(f" Inciando análise do bag: {os.path.basename(bag_path)}")
    print(f"==================================================")

    telemetry_data = {'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [], 'u': [], 'v': [], 'w': []}
    trajectory_data = {'time': [], 'x_ref': [], 'y_ref': [], 'z_ref': [], 'yaw_ref': [], 'u_ref': [], 'v_ref': [], 'w_ref': []}

    print("Extraindo mensagens do banco de dados...")
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            time_sec = timestamp / 1e9 
            
            if connection.topic == '/drone_simulator/telemetry_topic':
                telemetry_data['time'].append(time_sec)
                telemetry_data['x'].append(msg.pose.pose.position.x)
                telemetry_data['y'].append(msg.pose.pose.position.y)
                telemetry_data['z'].append(msg.pose.pose.position.z)
                telemetry_data['u'].append(msg.twist.twist.linear.x)
                telemetry_data['v'].append(msg.twist.twist.linear.y)
                telemetry_data['w'].append(msg.twist.twist.linear.z)
                
                q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                r, p, y = Rotation.from_quat(q).as_euler('xyz', degrees=True)
                telemetry_data['roll'].append(r)
                telemetry_data['pitch'].append(p)
                telemetry_data['yaw'].append(y)

            elif connection.topic == '/drone_simulator/trajectory_topic':
                trajectory_data['time'].append(time_sec)
                trajectory_data['x_ref'].append(msg.pose.pose.position.x)
                trajectory_data['y_ref'].append(msg.pose.pose.position.y)
                trajectory_data['z_ref'].append(msg.pose.pose.position.z)
                trajectory_data['u_ref'].append(msg.twist.twist.linear.x)
                trajectory_data['v_ref'].append(msg.twist.twist.linear.y)
                trajectory_data['w_ref'].append(msg.twist.twist.linear.z)

                q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                _, _, y_ref = Rotation.from_quat(q).as_euler('xyz', degrees=True)
                trajectory_data['yaw_ref'].append(y_ref)

    df_tel = pd.DataFrame(telemetry_data)
    df_traj = pd.DataFrame(trajectory_data)

    t_start = min(df_tel['time'].min(), df_traj['time'].min())
    df_tel['time'] = df_tel['time'] - t_start
    df_traj['time'] = df_traj['time'] - t_start

    results_dir = os.path.join(bag_path, "resultados")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Salvando arquivos CSV em: {results_dir}")
    df_tel.to_csv(os.path.join(results_dir, 'telemetria.csv'), index=False)
    df_traj.to_csv(os.path.join(results_dir, 'trajetoria_referencia.csv'), index=False)

    # 4. Geração de Gráficos
    print("Gerando gráficos de análise...")
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 9))

    # Gráfico 1: 3D Trajectory (AGORA COM .to_numpy())
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(df_traj['x_ref'].to_numpy(), df_traj['y_ref'].to_numpy(), df_traj['z_ref'].to_numpy(), label='Referência (Azul)', linestyle='--', color='blue', linewidth=2)
    ax1.plot(df_tel['x'].to_numpy(), df_tel['y'].to_numpy(), df_tel['z'].to_numpy(), label='Real (Vermelho)', color='red', alpha=0.8)
    ax1.scatter(df_traj['x_ref'].iloc[0], df_traj['y_ref'].iloc[0], df_traj['z_ref'].iloc[0], color='green', s=100, label='Início')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Trajetória 3D (Ref vs Real)')
    ax1.legend()

    # Gráficos Laterais: Rastreamento X, Y, Z ao longo do tempo (AGORA COM .to_numpy())
    axes = [fig.add_subplot(3, 2, 2), fig.add_subplot(3, 2, 4), fig.add_subplot(3, 2, 6)]
    labels = [('x_ref', 'x', 'Posição X (m)'), ('y_ref', 'y', 'Posição Y (m)'), ('z_ref', 'z', 'Posição Z (m)')]

    for ax, (col_ref, col_real, ylabel) in zip(axes, labels):
        ax.plot(df_traj['time'].to_numpy(), df_traj[col_ref].to_numpy(), 'b--', label='Ref')
        ax.plot(df_tel['time'].to_numpy(), df_tel[col_real].to_numpy(), 'r-', label='Real')
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
    
    axes[-1].set_xlabel('Tempo (s)')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'rastreamento_posicao.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"Gráfico salvo com sucesso: {plot_path}")
    print("Processamento concluído com sucesso!")

if __name__ == '__main__':
    workspace_bags_dir = os.path.join(os.getcwd(), 'src', 'm300_sim', 'bags')
    try:
        latest_bag = get_latest_bag(workspace_bags_dir)
        process_bag(latest_bag)
    except Exception as e:
        print(f"Erro ao processar dados: {e}")