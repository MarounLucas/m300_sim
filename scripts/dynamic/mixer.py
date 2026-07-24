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
        
        # Memory pre-allocation for high-frequency loops
        self.w_sq = np.zeros(4)
        self.w_cmds = np.zeros(4)

    def compute_motor_speed(self, u: np.ndarray) -> np.ndarray:
        """Maps the control action vector to target motor speeds.
        
        Args:
            u (np.ndarray): Array of desired virtual commands [u1, u2, u3, u4], where:
                u[0] = Total thrust (N)
                u[1] = Roll moment (N.m)
                u[2] = Pitch moment (N.m)
                u[3] = Yaw moment (N.m)
        
        Returns:
            np.ndarray: Array containing the reference angular velocities for
                the 4 motors in rad/s, bounded by the aircraft's physical limits.
        """
        u1, u2, u3, u4 = u[0], u[1], u[2], u[3]
        
        # Calculate squared angular velocities for each motor
        self.w_sq[0] = (
            (u1 / self.den_u1) + (u2 / self.den_u2) + 
            (u3 / self.den_u3) - (u4 * self.num_u4_bw / self.den_u4)
        )
        self.w_sq[1] = (
            (u1 / self.den_u1) - (u2 / self.den_u2) + 
            (u3 / self.den_u3) + (u4 * self.num_u4_bw / self.den_u4)
        )
        self.w_sq[2] = (
            (u1 / self.den_u1) - (u2 / self.den_u2) - 
            (u3 / self.den_u3) - (u4 * self.num_u4_fw / self.den_u4)
        )
        self.w_sq[3] = (
            (u1 / self.den_u1) + (u2 / self.den_u2) - 
            (u3 / self.den_u3) + (u4 * self.num_u4_fw / self.den_u4)
        )
        
        # Prevent negative values before square root calculation
        np.maximum(self.w_sq, 0.0, out=self.w_sq)
        
        # Calculate actual angular velocities
        np.sqrt(self.w_sq, out=self.w_cmds)
        
        # Apply saturation limits based on motor constraints
        np.clip(self.w_cmds, self.w_min, self.w_max, out=self.w_cmds)
        
        return self.w_cmds