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
    # Atualizado para buscar os bags do voo UNIFICADO
    bag_folders = glob.glob(os.path.join(bags_dir, 'voo_unificado_*'))
    if not bag_folders:
        raise FileNotFoundError(f"Nenhum bag encontrado na pasta: {bags_dir}")
    return max(bag_folders, key=os.path.getmtime)

def process_bag(bag_path: str):
    print(f"==================================================")
    print(f" Iniciando análise do bag UNIFICADO: {os.path.basename(bag_path)}")
    print(f"==================================================")

    telemetry_data = {'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [], 'u': [], 'v': [], 'w': []}
    trajectory_data = {'time': [], 'x_ref': [], 'y_ref': [], 'z_ref': [], 'yaw_ref': [], 'u_ref': [], 'v_ref': [], 'w_ref': []}

    print("A extrair mensagens do banco de dados...")
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            time_sec = timestamp / 1e9 
            
            # --- Tópico de Telemetria (Estado Real do Drone) ---
            if connection.topic == '/m300_sim/telemetry_topic':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
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

            # --- Tópico de Trajetória (Estado Desejado / Referência) ---
            elif connection.topic == '/m300_sim/trajectory_topic':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
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

    if df_tel.empty:
        print("ERRO: Nenhum dado de telemetria encontrado no bag!")
        return

    # Normalizar o tempo para começar do zero
    t_start = df_tel['time'].min()
    if not df_traj.empty:
        t_start = min(t_start, df_traj['time'].min())
        df_traj['time'] = df_traj['time'] - t_start
    
    df_tel['time'] = df_tel['time'] - t_start

    results_dir = os.path.join(bag_path, "resultados")
    os.makedirs(results_dir, exist_ok=True)

    print(f"A guardar ficheiros CSV em: {results_dir}")
    df_tel.to_csv(os.path.join(results_dir, 'telemetria_unificada.csv'), index=False)
    if not df_traj.empty:
        df_traj.to_csv(os.path.join(results_dir, 'trajetoria_referencia.csv'), index=False)

    # =========================================================================
    # GERAÇÃO DO DASHBOARD DE ANÁLISE
    # =========================================================================
    print("A gerar gráficos de análise unificada...")
    plt.style.use('ggplot')
    # Figura grande para acomodar todos os subplots
    fig = plt.figure(figsize=(20, 12))

    # Gráfico 1: Trajetória 3D (Ref vs Real)
    ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d') # Ocupa a coluna da esquerda toda
    if not df_traj.empty:
        ax1.plot(df_traj['x_ref'].to_numpy(), df_traj['y_ref'].to_numpy(), df_traj['z_ref'].to_numpy(), label='Referência (Azul)', linestyle='--', color='blue', linewidth=2)
    ax1.plot(df_tel['x'].to_numpy(), df_tel['y'].to_numpy(), df_tel['z'].to_numpy(), label='Real (Vermelho)', color='red', alpha=0.8)
    ax1.scatter(df_tel['x'].iloc[0], df_tel['y'].iloc[0], df_tel['z'].iloc[0], color='green', s=100, label='Início')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Trajetória 3D (Ref vs Real)')
    ax1.legend()

    # Gráfico 2: Rastreamento XYZ no tempo
    ax2 = fig.add_subplot(2, 3, 2)
    if not df_traj.empty:
        ax2.plot(df_traj['time'].to_numpy(), df_traj['x_ref'].to_numpy(), 'b--', alpha=0.5)
        ax2.plot(df_traj['time'].to_numpy(), df_traj['y_ref'].to_numpy(), 'g--', alpha=0.5)
        ax2.plot(df_traj['time'].to_numpy(), df_traj['z_ref'].to_numpy(), 'm--', alpha=0.5)
    ax2.plot(df_tel['time'].to_numpy(), df_tel['x'].to_numpy(), 'b-', label='X Real')
    ax2.plot(df_tel['time'].to_numpy(), df_tel['y'].to_numpy(), 'g-', label='Y Real')
    ax2.plot(df_tel['time'].to_numpy(), df_tel['z'].to_numpy(), 'm-', label='Z Real')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Posição (m)')
    ax2.set_title('Posições XYZ ao longo do tempo')
    ax2.legend()

    # Gráfico 3: Rastreamento de Yaw no tempo
    ax3 = fig.add_subplot(2, 3, 3)
    if not df_traj.empty:
        ax3.plot(df_traj['time'].to_numpy(), df_traj['yaw_ref'].to_numpy(), 'b--', label='Yaw Ref')
    ax3.plot(df_tel['time'].to_numpy(), df_tel['yaw'].to_numpy(), 'r-', label='Yaw Real')
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Ângulo (graus)')
    ax3.set_title('Rastreamento de Orientação (Yaw)')
    ax3.legend()

    # Gráfico 4: Atitude (Roll, Pitch) - Fundamental para ver solavancos na transição
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(df_tel['time'].to_numpy(), df_tel['roll'].to_numpy(), label='Roll (Inclinação Lat.)', color='red')
    ax4.plot(df_tel['time'].to_numpy(), df_tel['pitch'].to_numpy(), label='Pitch (Inclinação Long.)', color='green')
    ax4.set_xlabel('Tempo (s)')
    ax4.set_ylabel('Ângulo (graus)')
    ax4.set_title('Esforço de Atitude (Verificar Suavidade)')
    ax4.legend()

    # Gráfico 5: Velocidades Lineares (u, v, w)
    ax5 = fig.add_subplot(2, 3, 6)
    if not df_traj.empty:
        ax5.plot(df_traj['time'].to_numpy(), df_traj['u_ref'].to_numpy(), 'r--', alpha=0.3)
        ax5.plot(df_traj['time'].to_numpy(), df_traj['w_ref'].to_numpy(), 'b--', alpha=0.3)
    ax5.plot(df_tel['time'].to_numpy(), df_tel['u'].to_numpy(), label='u (vel x)', color='red')
    ax5.plot(df_tel['time'].to_numpy(), df_tel['v'].to_numpy(), label='v (vel y)', color='green')
    ax5.plot(df_tel['time'].to_numpy(), df_tel['w'].to_numpy(), label='w (vel z)', color='blue')
    ax5.set_xlabel('Tempo (s)')
    ax5.set_ylabel('Velocidade (m/s)')
    ax5.set_title('Perfil de Velocidade Linear')
    ax5.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'analise_voo_dashboard.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"Dashboard gráfico salvo com sucesso em: {plot_path}")
    print("Processamento concluído!")

if __name__ == '__main__':
    workspace_bags_dir = os.path.join(os.getcwd(), 'src', 'm300_sim', 'bags')
    try:
        latest_bag = get_latest_bag(workspace_bags_dir)
        process_bag(latest_bag)
    except Exception as e:
        print(f"Erro ao processar dados: {e}")