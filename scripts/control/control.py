'''
Descrição:
    Módulo responsável pela implementação do controlador em cascata para 
    o quadcoptero, inspirado, fortemente, na arquitetura de controle da PX4. 
    O sistema realiza o controle dividindo-se em malhas de: 
    posição (P), velocidade (PID), atitude/ângulo (P) e taxa de giro (PID).

Autor: Lucas Maroun de Almeida
'''

import numpy as np
import math
from typing import Type, Tuple

class CascadeController:
    '''
    Descrição:
        Controlador em Cascata baseado na formulação da PX4. Os controles 
        são divididos em blocos sucessivos: posição (P), velocidade (PID), 
        ângulo (P) e taxa de giro (PID).
        
        A diferença principal para a PX4 original é a possibilidade de operar
        em uma mesma frequência (1000hz) ou em frequências distintas.
    
    Métodos:
        __init__: Inicialização de parâmetros, ganhos e limites utilizados no controle.
        update_state: Recebe e atualiza o estado atual do drone (telemetria).
        desire_state: Recebe e define o estado desejado (waypoint) da trajetória.
        control: Coordena e executa todas as malhas de controle em 1000hz.
        control_px4: Coordena e executa todas as malhas de controle em frequencias distintas.
        _xy_pos_control: Realiza o controle proporcional de posição no plano XY.
        _z_pos_control: Realiza o controle proporcional de altitude (Z).
        _xy_vel_control: Realiza o controle PID da velocidade horizontal.
        _z_vel_control: Realiza o controle PID da velocidade vertical.
        _acel_to_atitude: Converte as acelerações desejadas em referências de atitude.
        _angle_control: Realiza o controle proporcional dos ângulos de Euler.
        _angular_rate_control: Realiza o controle PID das taxas de giro (ômega).
    '''
    
    def __init__(self, ac_model:Type):
        '''
        Descrição:
            Inicializa o controlador carregando os ganhos, variáveis de estado 
            e parâmetros físicos da aeronave.

        Args:
            ac_model (Type): Objeto contendo os parâmetros físicos e limites 
                               do modelo da aeronave.
            self.dt (float): Passo de tempo base da malha de controle.
        '''

        # Estado Atual
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)
        self.omega = np.zeros(3)

        # Estado Desejado
        self.des_position = np.zeros(3)
        self.des_velocity = np.zeros(3) 
        self.des_velocity_ff = np.zeros(3) 
        self.des_omega = np.zeros(3)
        self.des_angles = np.zeros(3)
        

        self.a_xy = np.zeros(2)
        self.a_z = 0.0

        self.t_total = 0

        # Parametros do integrativo
        self.rate_int_error = np.zeros(3)
        self.xy_vel_int_error = np.zeros(2)
        self.z_vel_int_error = 0.0
        self.max_integral = 2.0  # Evitar wind-up
        self.rate_max_integral = 0.5  # Evitar wind-up
        
        # Parametros do derivativo
        self.prev_omega = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.filtered_rate_derivative = np.zeros(3)
        self.filtered_xy_vel_derivative = np.zeros(2)
        self.filtered_z_vel_derivative = 0.0

        # ===== Ganhos: Taxa de Giro (PID) ===== #
        self.kp_rate = np.array([3.0, 3.0, 2.5]) 
        self.ki_rate = np.array([0.5, 0.5, 0.5]) # Reduzido para evitar sobrecarga
        self.kd_rate = np.array([0.15, 0.15, 0.1])
        
        # ===== Ganhos: Atitude (P) ===== #
        self.kp_att = np.array([4.0, 4.0, 2.5])
        
        # ===== Ganhos: Velocidade XY (PID) ===== #
        self.kp_xy_vel = np.array([2.0, 2.0]) 
        self.ki_xy_vel = np.array([0.15, 0.15]) # Praticamente zero
        self.kd_xy_vel = np.array([0.05, 0.05])
        
        # ===== Ganhos: Velocidade Z (PID) ===== #
        self.kp_z_vel = 2.0
        self.ki_z_vel = 0.05
        self.kd_z_vel = 0.4

        # ===== Ganhos: Posição XY e Z (P) ===== #
        # Como o FF está ativo, podemos deixar a malha de posição mais reativa
        self.kp_xy_pos = np.array([1.0, 1.0])
        self.kp_z_pos = 1.5
        
        # ===== Parametros da Aeronave ===== #
        self.mass = ac_model.mass
        self.min_z_vel = ac_model.max_ascent_speed
        self.max_z_vel = ac_model.max_descent_speed
        self.cruise_vel = ac_model.cruise_speed
        self.max_tilt = ac_model.max_tilt_angle
        self.max_roll_pitch_rate = ac_model.max_roll_pitch_rate
        self.max_yaw_rate = ac_model.max_yaw_rate
        self.max_rate = np.array([
            self.max_roll_pitch_rate,
            self.max_roll_pitch_rate,
            self.max_yaw_rate
        ])
        
    def update_state(self, state: np.ndarray) -> None:
        '''
        Descrição: 
            Recebe e atualiza o vetor de estado atual da aeronave.

        Args:
            state (np.ndarray): Vetor de estado atual contendo velocidades lineares (0:3), 
                                taxas angulares (3:6), ângulos de Euler (6:9) e 
                                posições XYZ (9:12).
        '''
        self.velocity = state[0:3]
        self.omega = state[3:6]
        self.angles = state[6:9]
        self.position = state[9:12]

    def desired_state(self, trajectory: np.ndarray, vel_ff:np.ndarray) -> None:
        '''
        Descrição: 
            Recebe o estado desejado da aeronave e instancia as referências 
            necessárias para as malhas de controle baseadas no waypoint.

        Args:
            trajectory (np.ndarray): Vetor contendo o waypoint atual da trajetória. 
                                     Contém posições desejadas (x, y, z) e o ângulo de yaw.
        '''
        self.des_position[:] = trajectory[0:3]
        self.des_angles[2] = trajectory[3]

        self.des_velocity_ff.fill(0.0)
    def control(self) -> Tuple[float, np.ndarray]:      
        '''
        Descrição:
            Método responsável por executar todas as malhas do controle em cascata.
            Chama sequencialmente os controladores de posição, velocidade, atitude 
            e taxa de giro.

        Returns:
            u_total (float): Empuxo total requisitado.
            tau (np.ndarray): Vetor de torques tridimensionais (Roll, Pitch, Yaw).
        '''
        self._xy_pos_control()
        self._z_pos_control()

        self._xy_vel_control()
        self._z_vel_control()

        u_total = self._acel_to_atitude()

        self._angle_control()

        tau = self._angular_rate_control()
        
        return u_total, tau
    
    def control_px4(self, tick:int) -> Tuple[float, np.ndarray]:      
        '''
        Descrição:
            Método responsável por executar todas as malhas do controle em cascata.
            Chama sequencialmente os controladores de posição, velocidade, atitude 
            e taxa de giro.

        Returns:
            u_total (float): Empuxo total requisitado.
            tau (np.ndarray): Vetor de torques tridimensionais (Roll, Pitch, Yaw).
        '''
        if tick % 20 == 0:
            self._xy_pos_control()
            self._z_pos_control()

            self._xy_vel_control(dt=0.02) 
            self._z_vel_control(dt=0.02)

            self.t_total = self._acel_to_atitude()

        if tick % 4 == 0:
            self._angle_control()
        

        tau = self._angular_rate_control(dt=0.001)

        return self.t_total, tau
    

    def _xy_pos_control(self) -> None:
        '''
        Descrição:
            Recebe a posição horizontal desejada e calcula a velocidade
            necessária para alcançá-la, limitando à velocidade de cruzeiro.
         
            Controle: Proporcional (P)
        '''
        error = self.des_position[0:2] - self.position[0:2]
        
        xy_vel = self.kp_xy_pos * error
        
        # norm_horizontal = np.linalg.norm(xy_vel)
        norm_horizontal = math.hypot(xy_vel[0], xy_vel[1])
        if norm_horizontal > self.cruise_vel:
            xy_vel = (xy_vel / norm_horizontal) * self.cruise_vel
            
        psi = self.angles[2] 
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        u_cmd = xy_vel[0] * c_psi + xy_vel[1] * s_psi
        v_cmd = -xy_vel[0] * s_psi + xy_vel[1] * c_psi
        
        u_ff = self.des_velocity_ff[0] * c_psi + self.des_velocity_ff[1] * s_psi
        v_ff = -self.des_velocity_ff[0] * s_psi + self.des_velocity_ff[1] * c_psi
        
        self.des_velocity[0] = u_cmd 
        self.des_velocity[1] = v_cmd 
          
    def _z_pos_control(self) -> None:
        '''
        Descrição:
            Recebe a posição vertical desejada e calcula a velocidade
            necessária para alcançá-la, limitando por subida/descida.
         
            Controle: Proporcional (P)
        '''
        error = self.des_position[2] - self.position[2]
        
        z_vel = self.kp_z_pos * error
        
        # self.des_velocity[2] = np.clip(z_vel, self.min_z_vel, self.max_z_vel)
        self.des_velocity[2] = max(self.min_z_vel, min(z_vel,self.max_z_vel))
       
    def _z_vel_control(self, dt:float) -> None:
        '''
        Descrição:
            Recebe a velocidade vertical desejada e calcula a aceleração
            necessária para atingi-la.
         
            Controle: Proporcional, Integral e Derivativo (PID)

        '''
        error = self.des_velocity[2] - self.velocity[2]
        
        self.z_vel_int_error = self.z_vel_int_error + (error * dt)
        # self.z_vel_int_error = np.clip(self.z_vel_int_error, -self.max_integral, self.max_integral)
        self.z_vel_int_error = max(-self.max_integral, min(self.z_vel_int_error, self.max_integral))
        
        raw_derivative = -(self.velocity[2] - self.prev_vel[2]) / dt
        
        alpha_vel = 0.2
        self.filtered_z_vel_derivative = (alpha_vel * raw_derivative) + ((1.0 - alpha_vel) * self.filtered_z_vel_derivative)
        
        self.a_z = (self.kp_z_vel * error) + \
              (self.ki_z_vel * self.z_vel_int_error) + \
              (self.kd_z_vel * self.filtered_z_vel_derivative)
        
        self.prev_vel[2] = self.velocity[2]
           
    def _xy_vel_control(self, dt:float) -> None:
        '''
        Descrição:
            Recebe as velocidades horizontais desejadas e calcula as acelerações
            necessárias para atingi-las.
         
            Controle: Proporcional, Integral e Derivativo (PID)
        '''
        error = self.des_velocity[0:2] - self.velocity[0:2]
        
        self.xy_vel_int_error = self.xy_vel_int_error + (error * dt)
        self.xy_vel_int_error = np.clip(self.xy_vel_int_error, -self.max_integral, self.max_integral)
        
        raw_derivative = -(self.velocity[0:2] - self.prev_vel[0:2]) / dt
        
        alpha_vel = 0.02 
        self.filtered_xy_vel_derivative = (alpha_vel * raw_derivative) + ((1.0 - alpha_vel) * self.filtered_xy_vel_derivative)
        
        self.a_xy = (self.kp_xy_vel * error) + \
              (self.ki_xy_vel * self.xy_vel_int_error) + \
              (self.kd_xy_vel * self.filtered_xy_vel_derivative)
        
        
        self.prev_vel[0:2] = self.velocity[0:2]

    def _acel_to_atitude(self) -> float:
        '''
        Descrição: 
            Converte as acelerações desejadas (a_xy, a_z) em referências de atitude 
            (roll e pitch) e calcula o empuxo total necessário para equilibrar as 
            forças.

        Returns:
            u_total (float): O valor de empuxo total calculado.
        '''
        g = 9.81
              
        des_pitch = np.arcsin(-self.a_xy[0] / g)
        des_roll = np.arcsin(self.a_xy[1] / g)   
               
        self.des_angles[1] = max(-self.max_tilt, min(des_pitch, self.max_tilt))
        self.des_angles[0] = max(-self.max_tilt, min(des_roll, self.max_tilt))
        
        roll_atual, pitch_atual = self.angles[0], self.angles[1]
        c_r, c_p = np.cos(roll_atual), np.cos(pitch_atual)
        
        u_total = self.mass * (9.81 - self.a_z) / (c_r * c_p)
        
        u_total = max(10.0, min(u_total, 180.0))
        
        return u_total
    
    def _angle_control(self) -> None:
        '''
        Descrição:
            Recebe a inclinação (angulos de Euler) desejada, calcula a taxa de giro 
            angular necessária para atingí-la.
         
            Controle: Proporcional (P)
        '''
        error = self.des_angles - self.angles
        
        omega_calc = self.kp_att * error
        
        omega_calc = np.clip(omega_calc, -self.max_rate, self.max_rate)
        
        self.des_omega[:] = omega_calc
     
    def _angular_rate_control(self, dt:float)-> np.ndarray:
        '''
        Descrição:
            Recebe a velocidade angular de giro desejada e calcula o torque 
            que os motores devem gerar para atingir o estado dinâmico. 
         
            Controle: PID

        Returns: 
            tau (np.ndarray): Vetor contendo os torques requisitados.
        '''
        error = self.des_omega - self.omega
        
        self.rate_int_error += (error * dt)
        self.rate_int_error = np.clip(self.rate_int_error, -self.rate_max_integral, self.rate_max_integral)
        
        raw_derivative = -(self.omega - self.prev_omega) / dt
        alpha_rate = 0.2 
        self.filtered_rate_derivative = (alpha_rate * raw_derivative) + ((1.0 - alpha_rate) * self.filtered_rate_derivative)

        tau = (self.kp_rate * error) + \
            (self.ki_rate * self.rate_int_error) + \
            (self.kd_rate * self.filtered_rate_derivative)
        
        self.prev_omega = self.omega
        
        return tau