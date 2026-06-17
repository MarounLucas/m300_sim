'''
Descrição: 
    Módulo responsável por converter as forças e momentos de controle 
    globais (Tração, Rolagem, Arfagem e Guinada) nas velocidades angulares de 
    referência individuais para cada motor.

Autor: Lucas Maroun de Almeida
'''

# ===== Importações ===== #
import numpy as np
from typing import Type    

class Mixer:
    '''
    Descrição:
        Classe responsável por realizar a mixagem de sinais de controle. 
        Pré-calcula as constantes da matriz inversa de alocação no momento 
        da inicialização.
    
    Métodos:
        __init__: Inicializa a classe e pré-calcula as constantes geométricas e 
         aerodinâmicas.
        
        compute_motor_speed: Mapeia o vetor de comandos (U) para as velocidades dos 
         motores (omega).
    '''
    def __init__(self, ac_model: Type):
        '''
        Descrição
            Inicializa o mixer e calcula, previamente, os denominadores das equações
            de mixagem.
        
        Args:
            ac_model (Type): Objeto contendo os parâmetros físicos e limites 
                               do modelo da aeronave.
        '''
        
        # Parâmetros da aeronave
        k_t0 = ac_model.k_t0    # Coeficiente de empuxo/tração (Thrust coefficient)
        k_q0 = ac_model.k_q0    # Coeficiente de torque/arrasto (Torque coefficient)
        dx = ac_model.dx_arm    # Distância do c.m. aos propulsores no eixo X
        dy_fw = ac_model.dy_fw  # Distância do c.m. aos propulsores frontais no eixo y
        dy_bw = ac_model.dy_bw  # Distância do c.m. aos propulsores traseiro no eixo y
        
        # Limites físicos dos motores
        self.w_min = ac_model.omega_min
        self.w_max = ac_model.omega_max
        
        # Denominadores das equações
        self.den_u1 = 4 * k_t0
        self.den_u2 = 2 * k_t0 * (dy_fw + dy_bw)
        self.den_u3 = 4 * k_t0 * dx
        self.den_u4 = 2 * k_q0 * (dy_fw + dy_bw)
        
        self.num_u4_fw = dy_fw
        self.num_u4_bw = dy_bw
        
        self.w_sq = np.zeros(4)
        self.w_cmds = np.zeros(4)

    def compute_motor_speed(self, u: np.ndarray) -> np.ndarray:
        """
        Recebe o vetor de ação u [u1, u2, u3, u4] e retorna as velocidades alvo 
        dos motores.
        
        Args:
            u (np.ndarray): Vetor de comandos virtuais desejados, onde:
                            u[0] = Tração total (N)
                            u[1] = Momento de Rolagem (N.m)
                            u[2] = Momento de Arfagem (N.m)
                            u[3] = Momento de Guinada (N.m)
        
        Return:
            w_cmds (np.ndarray): Vetor com as velocidades angulares de referência 
                                 para os 4 motores em rad/s, limitadas pelas 
                                 restrições físicas da aeronave.
            
        """
        u1, u2, u3, u4 = u[0], u[1], u[2], u[3]
        
        self.w_sq[0] = (u1/self.den_u1) + (u2/self.den_u2) + (u3/self.den_u3) - (u4 * self.num_u4_bw / self.den_u4)
        self.w_sq[1] = (u1/self.den_u1) - (u2/self.den_u2) + (u3/self.den_u3) + (u4 * self.num_u4_bw / self.den_u4)
        self.w_sq[2] = (u1/self.den_u1) - (u2/self.den_u2) - (u3/self.den_u3) - (u4 * self.num_u4_fw / self.den_u4)
        self.w_sq[3] = (u1/self.den_u1) + (u2/self.den_u2) - (u3/self.den_u3) + (u4 * self.num_u4_fw / self.den_u4)
        
        np.maximum(self.w_sq, 0.0, out=self.w_sq)
        np.sqrt(self.w_sq, out=self.w_cmds)
        np.clip(self.w_cmds, self.w_min, self.w_max, out=self.w_cmds)
        
        return self.w_cmds



