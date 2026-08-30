"""UAV control allocation mixer module.

Converts global control forces and moments (Thrust, Roll, Pitch, and Yaw)
into individual reference angular velocities for each motor.

Author: Lucas Maroun de Almeida
"""

from typing import Any
import numpy as np


class Mixer:
    """Mixes control signals to determine individual motor speeds.

    Pre-calculates the constants of the inverse allocation matrix during
    initialization to optimize computation within the high-frequency control loop.
    """

    def __init__(self, ac_model: Any) -> None:
        """Initializes the mixer and pre-calculates equation denominators.

        Args:
            ac_model (Any): Object containing the aircraft's physical parameters
                and limits (e.g., thrust/torque coefficients, arm lengths).
        """
        # Aircraft parameters
        k_t0 = ac_model.k_t0      # Thrust coefficient
        k_q0 = ac_model.k_q0      # Torque coefficient
        dx = ac_model.dx_arm      # Distance from c.m. to motors on the X-axis
        dy_fw = ac_model.dy_fw    # Distance from c.m. to front motors on the Y-axis
        dy_bw = ac_model.dy_bw    # Distance from c.m. to rear motors on the Y-axis
        
        # Physical limits of the motors
        self.w_min = ac_model.omega_min
        self.w_max = ac_model.omega_max
        
        # Pre-calculated denominators for the mixing equations
        self.den_u1 = 4.0 * k_t0
        self.den_u2 = 2.0 * k_t0 * (dy_fw + dy_bw)
        self.den_u3 = 4.0 * k_t0 * dx
        self.den_u4 = 2.0 * k_q0 * (dy_fw + dy_bw)
        
        # Pre-calculated numerators for the yaw mixing terms
        self.num_u4_fw = dy_fw
        self.num_u4_bw = dy_bw

        # Init parameters
        self.w_sq = np.zeros(4)
        self.w_cmds = np.zeros(4)

    def compute_motor_speed(self, u: np.ndarray) -> np.ndarray:
        """Maps the control action vector to target motor speeds with Thrust Prioritization.
        """
        u1, u2, u3, u4 = u[0], u[1], u[2], u[3]
        
        w_sq = np.zeros(4)
        w_sq[0] = (u1 / self.den_u1) + (u2 / self.den_u2) + (u3 / self.den_u3) - (u4 * self.num_u4_bw / self.den_u4)
        w_sq[1] = (u1 / self.den_u1) - (u2 / self.den_u2) + (u3 / self.den_u3) + (u4 * self.num_u4_bw / self.den_u4)
        w_sq[2] = (u1 / self.den_u1) - (u2 / self.den_u2) - (u3 / self.den_u3) - (u4 * self.num_u4_fw / self.den_u4)
        w_sq[3] = (u1 / self.den_u1) + (u2 / self.den_u2) - (u3 / self.den_u3) + (u4 * self.num_u4_fw / self.den_u4)
        
        w_max_sq = self.w_max ** 2
        w_min_sq = self.w_min ** 2
        
        # Verify if the values clip into the limits
        max_req = np.max(w_sq)
        min_req = np.min(w_sq)
        
        if max_req > w_max_sq or min_req < w_min_sq:
            # Isola a parcela que é responsável apenas por manter o drone no ar (Empuxo)
            thrust_comp = u1 / self.den_u1
            
            # Isola a parcela responsável por girar o drone (Torques)
            torque_comp = w_sq - thrust_comp
            
            # Fator de escala para amassar os torques sem alterar o empuxo
            scale = 1.0
            
            # Se o limite superior estourou, descobre quanto precisa reduzir os torques
            if max_req > w_max_sq:
                # Evita divisão por zero caso o torque_comp seja minúsculo
                max_torque = np.max(np.abs(torque_comp)) + 1e-6 
                s_max = (w_max_sq - thrust_comp) / max_torque
                scale = min(scale, s_max)
                
            # Se o limite inferior estourou (motor ia parar)
            if min_req < w_min_sq:
                max_torque = np.max(np.abs(torque_comp)) + 1e-6
                s_min = (thrust_comp - w_min_sq) / max_torque
                scale = min(scale, s_min)
                
            # Garante que a escala não se torne negativa em casos extremos
            scale = max(0.0, scale) 
            
            # Reconstrói o comando: Empuxo intacto + Torques reduzidos
            w_sq = thrust_comp + (torque_comp * scale)
            
        # Proteção final por segurança
        np.clip(w_sq, w_min_sq, w_max_sq, out=w_sq)
        
        np.sqrt(w_sq, out=self.w_cmds)
        
        return self.w_cmds