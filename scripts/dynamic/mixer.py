"""UAV control allocation mixer module.

Convert collective thrust and body torque commands into individual reference
angular velocities for each rotor, prioritizing collective thrust under
actuator saturation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Mixer:
    """Allocate total thrust and body torques to individual rotor speeds.

    The inverse allocation constants are pre-computed at initialization so the
    high-frequency loop only performs basic arithmetic and saturation scaling.

    Attributes:
        w_min: Minimum rotor angular velocity [rad/s].
        w_max: Maximum rotor angular velocity [rad/s].
    """

    def __init__(self, ac_model: Any) -> None:
        """Pre-compute inverse allocation denominators from aircraft geometry."""
        k_t0 = ac_model.k_t0
        k_q0 = ac_model.k_q0
        dx = ac_model.dx_arm
        dy_fw = ac_model.dy_fw
        dy_bw = ac_model.dy_bw

        self.w_min = ac_model.omega_min
        self.w_max = ac_model.omega_max

        self.den_u1 = 4.0 * k_t0
        self.den_u2 = 2.0 * k_t0 * (dy_fw + dy_bw)
        self.den_u3 = 4.0 * k_t0 * dx
        self.den_u4 = 2.0 * k_q0 * (dy_fw + dy_bw)

        self.num_u4_fw = dy_fw
        self.num_u4_bw = dy_bw

        self.w_sq = np.zeros(4)
        self.w_cmds = np.zeros(4)

    def compute_motor_speed(self, u: np.ndarray) -> np.ndarray:
        """Convert control wrench to rotor speeds, prioritizing thrust on saturation.

        When a requested rotor speed exceeds ``[w_min, w_max]``, roll/pitch/yaw
        torque contributions are scaled down uniformly while keeping collective
        thrust unchanged.

        Args:
            u: Control wrench ``[Fz, Tx, Ty, Tz]`` in the body frame, where
                ``Fz`` is total thrust [N] and ``Tx, Ty, Tz`` are torques [N·m].

        Returns:
            Target rotor angular velocities ``[w1, w2, w3, w4]`` [rad/s].
        """
        u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

        c_u1 = u1 / self.den_u1
        c_u2 = u2 / self.den_u2
        c_u3 = u3 / self.den_u3
        c_u4_bw = u4 * self.num_u4_bw / self.den_u4
        c_u4_fw = u4 * self.num_u4_fw / self.den_u4

        w_sq = np.zeros(4)
        w_sq[0] = c_u1 + c_u2 + c_u3 - c_u4_bw
        w_sq[1] = c_u1 - c_u2 + c_u3 + c_u4_bw
        w_sq[2] = c_u1 - c_u2 - c_u3 - c_u4_fw
        w_sq[3] = c_u1 + c_u2 - c_u3 + c_u4_fw

        w_max_sq = self.w_max ** 2
        w_min_sq = self.w_min ** 2

        max_req = np.max(w_sq)
        min_req = np.min(w_sq)

        if max_req > w_max_sq or min_req < w_min_sq:
            # Split collective thrust from differential torque contributions so
            # attitude authority can be scaled down without losing lift.
            thrust_comp = c_u1
            torque_comp = w_sq - thrust_comp
            scale = 1.0

            if max_req > w_max_sq:
                max_torque = np.max(np.abs(torque_comp)) + 1e-6
                s_max = (w_max_sq - thrust_comp) / max_torque
                scale = min(scale, s_max)

            if min_req < w_min_sq:
                max_torque = np.max(np.abs(torque_comp)) + 1e-6
                s_min = (thrust_comp - w_min_sq) / max_torque
                scale = min(scale, s_min)

            scale = max(0.0, scale)
            w_sq = thrust_comp + (torque_comp * scale)

        np.clip(w_sq, w_min_sq, w_max_sq, out=w_sq)
        np.sqrt(w_sq, out=self.w_cmds)

        return self.w_cmds
