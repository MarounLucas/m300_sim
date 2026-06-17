"""
Descrição: 
    Módulo contendo as equações dinâmicas de corpo rígido com 6 graus de 
    liberdade (6-DoF) para simulação da aeronave.

Autor: Lucas Maroun de Almeida
"""

import numpy as np
# from numba import njit

# Constante global
GRAVITY: float = 9.81  # Aceleração da gravidade em m/s^2

# @njit
def six_dof_model(state: np.ndarray, ac_params: np.ndarray,
                  wind_ned: np.ndarray, w_cmds: np.ndarray) -> np.ndarray:  
    """
    Calcula as derivadas do estado do quadcoptero baseadas no formalismo clássico
    de Newton-Euler.
        
    Args:
        state (np.ndarray): Vetor de estado atual:
            [0:3]   u, v, w         -> Velocidades lineares (Referencial do Corpo)
            [3:6]   p, q, r         -> Velocidades angulares (Referencial do Corpo)
            [6:9]   phi, theta, psi -> Ângulos de Euler (Referencial Inercial)
            [9:12]  px, py, pz      -> Posições lineares (Referencial Inercial)
            [12:20] w, alpha        -> Velocidades e Acelerações de cada motor
        ac_params (np.ndarray): Parâmetros físicos da aeronave.
        wind_ned (np.ndarray): Vetor do vento em coordenadas cartesianas NED [wx, wy, wz].
        w_cmds (np.ndarray): Velocidade angular de referência para cada motor [rad/s].
        
    Returns:
        np.ndarray: Vetor contendo as derivadas de estado.
    """
    dx = np.zeros(20)
    
    # Extração de variáveis de estado 
    u, v, w = state[0], state[1], state[2]
    p, q, r = state[3], state[4], state[5]
    phi, theta, psi = state[6], state[7], state[8]
    
    # Motores (Velocidade angular e aceleração angular)
    w_m1, alpha_m1 = state[12], state[13]
    w_m2, alpha_m2 = state[14], state[15]
    w_m3, alpha_m3 = state[16], state[17]
    w_m4, alpha_m4 = state[18], state[19]
    
    # Extração de parâmetros da aeronave
    mass, jx, jy, jz, jxz, cd, k_t0, k_q0, tau, kp, xi, dx_arm, dy_fw, dy_bw = ac_params
    
    # Pré-cálculo de operações trigonométricas
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta, t_theta = np.sin(theta), np.cos(theta), np.tan(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    
    
    # Gravidade decomposta no sistema de coordenadas do corpo
    gb_x = -GRAVITY * s_theta
    gb_y =  GRAVITY * s_phi * c_theta
    gb_z =  GRAVITY * c_phi * c_theta
    
    wx_ned, wy_ned, wz_ned = wind_ned[0], wind_ned[1], wind_ned[2]
    
    # Transformação do Vento (Inercial NED -> Corpo)
    wx_b = wx_ned * (c_theta * c_psi) + \
           wy_ned * (c_theta * s_psi) + \
           wz_ned * (-s_theta)
           
    wy_b = wx_ned * ((-c_phi * s_psi) + (s_phi * s_theta * c_psi)) + \
           wy_ned * ((c_phi * c_psi) + (s_phi * s_theta * s_psi)) + \
           wz_ned * (s_phi * c_theta)
           
    wz_b = wx_ned * ((s_phi * s_psi) + (c_phi * s_theta * c_psi)) + \
           wy_ned * ((-s_phi * c_psi) + (c_phi * s_theta * s_psi)) + \
           wz_ned * (c_phi * c_theta)
    
    # ===== Modelagem da Dinâmica dos Motores (1ª ordem) ===== #
    tau_sq = tau ** 2
    dx[12] = alpha_m1
    dx[13] = ((-2 * xi * tau * alpha_m1) - w_m1 + (kp * w_cmds[0])) / tau_sq
    
    dx[14] = alpha_m2
    dx[15] = ((-2 * xi * tau * alpha_m2) - w_m2 + (kp * w_cmds[1])) / tau_sq
    
    dx[16] = alpha_m3
    dx[17] = ((-2 * xi * tau * alpha_m3) - w_m3 + (kp * w_cmds[2])) / tau_sq
    
    dx[18] = alpha_m4
    dx[19] = ((-2 * xi * tau * alpha_m4) - w_m4 + (kp * w_cmds[3])) / tau_sq

    # Cálculo de Empuxo (T) e Torque (Q) gerado por cada motor real
    t1 = k_t0 * (w_m1 ** 2); q1 = k_q0 * (w_m1 ** 2)
    t2 = k_t0 * (w_m2 ** 2); q2 = k_q0 * (w_m2 ** 2)
    t3 = k_t0 * (w_m3 ** 2); q3 = k_q0 * (w_m3 ** 2)
    t4 = k_t0 * (w_m4 ** 2); q4 = k_q0 * (w_m4 ** 2)
    
    # ===== Forças Aerodinâmicas (Referencial do Corpo) ===== #
    fx = -cd * (u - wx_b)
    fy = -cd * (v - wy_b)
    fz = -(t1 + t2 + t3 + t4) - (cd * (w - wz_b)) 
    
    # ===== Momentos Externos (Referencial do Corpo) ===== #
    l = (t1 - t2) * dy_fw + (t4 - t3) * dy_bw
    m = (t1 + t2 - t3 - t4) * dx_arm
    n = -q1 + q2 - q3 + q4
    
    # ===== Equações de Movimento Translacional ===== #
    dx[0] = (fx / mass) + gb_x - (q * w) + (v * r)
    dx[1] = (fy / mass) + gb_y - (p * w) + (u * r)
    dx[2] = (fz / mass) + gb_z - (p * v) + (u * q)
    
    # ===== Equações de Movimento Rotacional ===== #
    den = (jx * jz) - (jxz ** 2)
    dx[3] = (1 / den) * ((jz * (jy - jz) - jxz**2) * q * r + jxz * (jx - jy + jz) * p * q + jz * l + jxz * n)
    dx[4] = (1 / jy) * (((jz - jx) * r * p) - (jxz * (p**2 - r**2)) + m) 
    dx[5] = (1 / den) * ((jx * (jx - jy) + jxz**2) * p * q + jxz * (jy - jx - jz) * q * r + jxz * l + jx * n)
    
    # ===== Cinemática de Euler ===== # 
    dx[6] = p + (q * s_phi * t_theta) + (r * c_phi * t_theta)
    dx[7] = (q * c_phi) - (r * s_phi)
    dx[8] = (q * s_phi / c_theta) + (r * c_phi / c_theta)
    
    # ===== Equações de Posição ===== #
    dx[9]  = u * (c_theta * c_psi) + \
             v * ((-c_phi * s_psi) + (s_phi * s_theta * c_psi)) + \
             w * ((s_phi * s_psi) + (c_phi * s_theta * c_psi))
             
    dx[10] = u * (c_theta * s_psi) + \
             v * ((c_phi * c_psi) + (s_phi * s_theta * s_psi)) + \
             w * ((-s_phi * c_psi) + (c_phi * s_theta * s_psi))
             
    dx[11] = u * (-s_theta) + \
             v * (s_phi * c_theta) + \
             w * (c_phi * c_theta)
    
    return dx