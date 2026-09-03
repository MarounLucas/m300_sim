import pandas as pd
import numpy as np
import os

def process_single_log(input_filepath, output_filepath, aircraft_params):
    df = pd.read_csv(input_filepath)

    # Parâmetros para a_nom    
    mass = aircraft_params['mass']
    jx = aircraft_params['jx']
    jy = aircraft_params['jy']
    jz = aircraft_params['jz']
    jxz = aircraft_params['jxz']
    g = aircraft_params.get('g', 9.80665)
    
    # ==== Filtragem ==== #
    # Verifica velocidade linear e Altitude
    v_linear = np.sqrt(df['Vel_u_ms']**2 + df['Vel_v_ms']**2 + df['Vel_w_ms']**2)
    mask_active = (df['Pos_Alt_m'] > 0.05) & (v_linear > 0.05)
    
    if not mask_active.any():
        print(f"  -> AVISO: Nenhum voo dinâmico detectado em {os.path.basename(input_filepath)}. Ignorando.")
        return False

    # Coleta de tempo inicial e final
    t_start_idx = mask_active.idxmax() # Primeiro tempo dentro da mask
    t_end_idx = mask_active[::-1].idxmax() # Ultimo tempo dentro da mask

    # df filtrado
    df_trimmed = df.loc[t_start_idx:t_end_idx].copy().reset_index(drop=True)


    # ==== Calculo Nominal ==== #
    phi = np.radians(df_trimmed['Roll_deg'])
    theta = np.radians(df_trimmed['Pitch_deg'])
    
    p = df_trimmed['Rate_p_rads']
    q = df_trimmed['Rate_q_rads']
    r = df_trimmed['Rate_r_rads']
    
    l = df_trimmed['Torque_Tx_Nm']
    m = df_trimmed['Torque_Ty_Nm']
    n = df_trimmed['Torque_Tz_Nm']
    
    fx = 0.0
    fy = 0.0
    fz = df_trimmed['Thrust_Fz_N']

    u = df_trimmed['Vel_u_ms']
    v = df_trimmed['Vel_v_ms']
    w = df_trimmed['Vel_w_ms']

    gb_x = -g * np.sin(theta)
    gb_y = g * np.cos(theta) * np.sin(phi)
    gb_z = g * np.cos(theta) * np.cos(phi)

    # ---------------------------------------------------------
    d_u_nom = (fx / mass) + gb_x - (q * w) + (v * r)
    d_v_nom = (fy / mass) + gb_y - (p * w) + (u * r)
    d_w_nom = -(fz / mass) + gb_z - (p * v) + (u * q)
    
    den = (jx * jz) - (jxz ** 2)
    
    d_p_nom = (1.0 / den) * (
        (jz * (jy - jz) - jxz**2) * q * r
        + jxz * (jx - jy + jz) * p * q
        + jz * l
        + jxz * n
    )
    
    d_q_nom = (1.0 / jy) * (
        ((jz - jx) * r * p) - (jxz * (p**2 - r**2)) + m
    )
    
    d_r_nom = (1.0 / den) * (
        (jx * (jx - jy) + jxz**2) * p * q
        + jxz * (jy - jx - jz) * q * r
        + jxz * l
        + jx * n
    )
    
    # Calculo Residual
    df_trimmed['Delta_a_u'] = df_trimmed['Accel_x_ms2'] - d_u_nom
    df_trimmed['Delta_a_v'] = df_trimmed['Accel_y_ms2'] - d_v_nom
    df_trimmed['Delta_a_w'] = df_trimmed['Accel_z_ms2'] - d_w_nom
    
    df_trimmed['Delta_alpha_p'] = df_trimmed['Alpha_p_rads2'] - d_p_nom
    df_trimmed['Delta_alpha_q'] = df_trimmed['Alpha_q_rads2'] - d_q_nom
    df_trimmed['Delta_alpha_r'] = df_trimmed['Alpha_r_rads2'] - d_r_nom

    # Normalização de Angulos
    for angle in ['Roll_deg', 'Pitch_deg', 'Yaw_deg']:
        angle_rad = np.radians(df_trimmed[angle])
        df_trimmed[f'sin_{angle}'] = np.sin(angle_rad)
        df_trimmed[f'cos_{angle}'] = np.cos(angle_rad)
        
    residual_cols = ['Delta_a_u', 'Delta_a_v', 'Delta_a_w', 'Delta_alpha_p', 'Delta_alpha_q', 'Delta_alpha_r']
    for col in residual_cols:
        std_val = df_trimmed[col].std()
        if std_val == 0: std_val = 1e-6
        df_trimmed[f'{col}_zscore'] = (df_trimmed[col] - df_trimmed[col].mean()) / std_val
        
    
    # Salva em CSV
    df_trimmed.to_csv(output_filepath, index=False)

    # Salva em TXT    
    # output_filepath_txt = os.path.splitext(output_filepath)[0] + '.txt'
    # df_trimmed.to_csv(output_filepath_txt, sep=';', index=False)
    
    return True


def process_directory(folder_path, aircraft_params):
    # Verifica se o diretório existe
    if not os.path.isdir(folder_path):
        print(f"Erro: O diretório especificado não existe: {folder_path}")
        return

    # Procura arquivos csv dentro do diretório
    # Procura arquivos que não comecem com 'ftr_'
    csv_files = [f for f in os.listdir(folder_path) 
                 if f.endswith('.csv') and not f.startswith('ftr_')]
    
    if not csv_files:
        print(f"Nenhum arquivo CSV válido para processar na pasta: {folder_path}")
        return


    sucessos = 0
    for filename in csv_files:
        input_filepath = os.path.join(folder_path, filename)
        output_filepath = os.path.join(folder_path, 'ftr', f"ftr_{filename}")
        
        try:
            sucesso = process_single_log(input_filepath, output_filepath, aircraft_params)
            if sucesso:
                sucessos += 1
        except Exception as e:
            print(f"  -> ERRO ao processar {filename}: {e}")
            
    print(f"\nProcessamento em lote concluído! {sucessos}/{len(csv_files)} arquivos processados com sucesso.")


if __name__ == "__main__":
    # Parametros do drone para a_nom
    m350_params = {
        "mass": 6.47,
        "jx": 0.214,
        "jy": 0.2,
        "jz": 0.389,
        "jxz": 0.0,       
        'g': 9.81        
    }
    
    logs_folder = '/workspace/src/m300_sim/data/flight_logs/sim'
    process_directory(logs_folder, m350_params)