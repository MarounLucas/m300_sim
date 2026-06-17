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
    # Atualizado para buscar os bags do voo ONLINE
    bag_folders = glob.glob(os.path.join(bags_dir, 'voo_online_*'))
    if not bag_folders:
        raise FileNotFoundError(f"Nenhum bag encontrado na pasta: {bags_dir}")
    return max(bag_folders, key=os.path.getmtime)

def process_bag(bag_path: str):
    print(f"==================================================")
    print(f" Iniciando análise do bag ONLINE: {os.path.basename(bag_path)}")
    print(f"==================================================")

    # Apenas telemetria, não temos trajetória de referência no voo manual
    telemetry_data = {'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [], 'u': [], 'v': [], 'w': []}

    print("Extraindo mensagens do banco de dados...")
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            
            # Buscando apenas o tópico de telemetria correto
            if connection.topic == '/m300_sim/telemetry_topic':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                time_sec = timestamp / 1e9 
                
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

    df_tel = pd.DataFrame(telemetry_data)
    
    # Tratamento de erro caso o bag não contenha o tópico de telemetria
    if df_tel.empty:
        print("ERRO: Nenhum dado de telemetria encontrado no bag!")
        return

    # Normalizando o tempo para começar do zero
    t_start = df_tel['time'].min()
    df_tel['time'] = df_tel['time'] - t_start

    results_dir = os.path.join(bag_path, "resultados")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Salvando arquivo CSV em: {results_dir}")
    df_tel.to_csv(os.path.join(results_dir, 'telemetria_online.csv'), index=False)

    # Geração de Gráficos
    print("Gerando gráficos de análise...")
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 10))

    # Gráfico 1: Trajetória 3D (Apenas percurso real)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(df_tel['x'].to_numpy(), df_tel['y'].to_numpy(), df_tel['z'].to_numpy(), label='Voo Manual (Vermelho)', color='red', alpha=0.8)
    ax1.scatter(df_tel['x'].iloc[0], df_tel['y'].iloc[0], df_tel['z'].iloc[0], color='green', s=100, label='Início')
    ax1.scatter(df_tel['x'].iloc[-1], df_tel['y'].iloc[-1], df_tel['z'].iloc[-1], color='black', s=100, marker='X', label='Fim')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Trajetória 3D (Voo Manual)')
    ax1.legend()

    # Gráfico 2: Posições X, Y, Z no tempo
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df_tel['time'].to_numpy(), df_tel['x'].to_numpy(), label='X', color='red')
    ax2.plot(df_tel['time'].to_numpy(), df_tel['y'].to_numpy(), label='Y', color='green')
    ax2.plot(df_tel['time'].to_numpy(), df_tel['z'].to_numpy(), label='Z', color='blue')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Posição (m)')
    ax2.set_title('Posições ao longo do tempo')
    ax2.legend()

    # Gráfico 3: Atitude (Roll, Pitch, Yaw)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(df_tel['time'].to_numpy(), df_tel['roll'].to_numpy(), label='Roll', color='red')
    ax3.plot(df_tel['time'].to_numpy(), df_tel['pitch'].to_numpy(), label='Pitch', color='green')
    ax3.plot(df_tel['time'].to_numpy(), df_tel['yaw'].to_numpy(), label='Yaw', color='blue')
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Ângulo (graus)')
    ax3.set_title('Atitude da Aeronave')
    ax3.legend()

    # Gráfico 4: Velocidades (u, v, w)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(df_tel['time'].to_numpy(), df_tel['u'].to_numpy(), label='u (vel x)', color='red')
    ax4.plot(df_tel['time'].to_numpy(), df_tel['v'].to_numpy(), label='v (vel y)', color='green')
    ax4.plot(df_tel['time'].to_numpy(), df_tel['w'].to_numpy(), label='w (vel z)', color='blue')
    ax4.set_xlabel('Tempo (s)')
    ax4.set_ylabel('Velocidade (m/s)')
    ax4.set_title('Velocidades Lineares')
    ax4.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'analise_voo_online.png')
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