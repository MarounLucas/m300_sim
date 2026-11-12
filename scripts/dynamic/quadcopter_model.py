"""6-DoF rigid-body and motor dynamics model for the M350 quadcopter.

Implements the Newton-Euler equations of motion in the body frame, second-order
actuator dynamics for the four rotors, aerodynamic linear drag in the presence
of inertial wind, and optional neural network residual acceleration injection.
"""

from __future__ import annotations

import numpy as np

GRAVITY: float = 9.81


def six_dof_model(
    state: np.ndarray,
    ac_params: np.ndarray,
    wind_ned: np.ndarray,
    w_cmds: np.ndarray,
    nn_residual: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the 20 state derivatives of the quadcopter using Newton-Euler equations.

    Args:
        state: Current 20-element state vector:
            - ``[0:3]`` ``u, v, w`` linear velocities [m/s] (body frame).
            - ``[3:6]`` ``p, q, r`` angular rates [rad/s] (body frame).
            - ``[6:9]`` ``phi, theta, psi`` Euler angles [rad] (NED frame).
            - ``[9:12]`` ``px, py, pz`` positions [m] (NED frame).
            - ``[12:20]`` ``w_m1, alpha_m1, ..., w_m4, alpha_m4`` rotor speeds [rad/s]
              and angular accelerations [rad/s²].
        ac_params: 14-element array of physical, inertia, and propulsion parameters.
        wind_ned: Instantaneous wind velocity vector ``[wx, wy, wz]`` [m/s] (NED frame).
        w_cmds: Commanded rotor speeds ``[w1, w2, w3, w4]`` [rad/s].
        nn_residual: Optional 6-element residual acceleration vector
            ``[Delta_a_u, Delta_a_v, Delta_a_w, Delta_alpha_p, Delta_alpha_q, Delta_alpha_r]``
            in ``[m/s²]`` and ``[rad/s²]`` (body frame) predicted by the MLP.

    Returns:
        20-element state derivative vector ``dx``.
    """
    dx = np.zeros(20, dtype=np.float64)

    u, v, w = state[0], state[1], state[2]
    p, q, r = state[3], state[4], state[5]
    phi, theta, psi = state[6], state[7], state[8]

    w_m1, alpha_m1 = state[12], state[13]
    w_m2, alpha_m2 = state[14], state[15]
    w_m3, alpha_m3 = state[16], state[17]
    w_m4, alpha_m4 = state[18], state[19]

    (
        mass,
        jx,
        jy,
        jz,
        jxz,
        cd,
        k_t0,
        k_q0,
        tau,
        kp,
        xi,
        dx_arm,
        dy_fw,
        dy_bw,
    ) = ac_params

    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta, t_theta = np.sin(theta), np.cos(theta), np.tan(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)

    # Gravity vector projected into the body frame
    gb_x = -GRAVITY * s_theta
    gb_y = GRAVITY * s_phi * c_theta
    gb_z = GRAVITY * c_phi * c_theta

    wx_ned, wy_ned, wz_ned = wind_ned[0], wind_ned[1], wind_ned[2]

    # Rotate wind velocity from inertial NED frame into body frame
    wx_b = (
        wx_ned * (c_theta * c_psi)
        + wy_ned * (c_theta * s_psi)
        - wz_ned * s_theta
    )
    wy_b = (
        wx_ned * ((-c_phi * s_psi) + (s_phi * s_theta * c_psi))
        + wy_ned * ((c_phi * c_psi) + (s_phi * s_theta * s_psi))
        + wz_ned * (s_phi * c_theta)
    )
    wz_b = (
        wx_ned * ((s_phi * s_psi) + (c_phi * s_theta * c_psi))
        + wy_ned * ((-s_phi * c_psi) + (c_phi * s_theta * s_psi))
        + wz_ned * (c_phi * c_theta)
    )

    # Second-order rotor actuator dynamics
    tau_sq = tau**2
    dx[12] = alpha_m1
    dx[13] = ((-2.0 * xi * tau * alpha_m1) - w_m1 + (kp * w_cmds[0])) / tau_sq

    dx[14] = alpha_m2
    dx[15] = ((-2.0 * xi * tau * alpha_m2) - w_m2 + (kp * w_cmds[1])) / tau_sq

    dx[16] = alpha_m3
    dx[17] = ((-2.0 * xi * tau * alpha_m3) - w_m3 + (kp * w_cmds[2])) / tau_sq

    dx[18] = alpha_m4
    dx[19] = ((-2.0 * xi * tau * alpha_m4) - w_m4 + (kp * w_cmds[3])) / tau_sq

    t1 = k_t0 * (w_m1**2)
    q1 = k_q0 * (w_m1**2)

    t2 = k_t0 * (w_m2**2)
    q2 = k_q0 * (w_m2**2)

    t3 = k_t0 * (w_m3**2)
    q3 = k_q0 * (w_m3**2)

    t4 = k_t0 * (w_m4**2)
    q4 = k_q0 * (w_m4**2)

    fx = -cd * (u - wx_b)
    fy = -cd * (v - wy_b)
    fz = -(t1 + t2 + t3 + t4) - (cd * (w - wz_b))

    tau_x = (t1 - t2) * dy_fw + (t4 - t3) * dy_bw
    tau_y = (t1 + t2 - t3 - t4) * dx_arm
    tau_z = -q1 + q2 - q3 + q4

    dx[0] = (fx / mass) + gb_x - (q * w) + (v * r)
    dx[1] = (fy / mass) + gb_y + (p * w) - (u * r)
    dx[2] = (fz / mass) + gb_z - (p * v) + (u * q)

    den = (jx * jz) - (jxz**2)
    dx[3] = (1.0 / den) * (
        (jz * (jy - jz) - jxz**2) * q * r
        + jxz * (jx - jy + jz) * p * q
        + jz * tau_x
        + jxz * tau_z
    )
    dx[4] = (1.0 / jy) * (
        ((jz - jx) * r * p) - (jxz * (p**2 - r**2)) + tau_y
    )
    dx[5] = (1.0 / den) * (
        (jx * (jx - jy) + jxz**2) * p * q
        + jxz * (jy - jx - jz) * q * r
        + jxz * tau_x
        + jx * tau_z
    )

    if nn_residual is not None:
        dx[0:6] += nn_residual[0:6]

    dx[6] = p + (q * s_phi * t_theta) + (r * c_phi * t_theta)
    dx[7] = (q * c_phi) - (r * s_phi)
    dx[8] = (q * s_phi / c_theta) + (r * c_phi / c_theta)

    dx[9] = (
        u * (c_theta * c_psi)
        + v * ((-c_phi * s_psi) + (s_phi * s_theta * c_psi))
        + w * ((s_phi * s_psi) + (c_phi * s_theta * c_psi))
    )
    dx[10] = (
        u * (c_theta * s_psi)
        + v * ((c_phi * c_psi) + (s_phi * s_theta * s_psi))
        + w * ((-s_phi * c_psi) + (c_phi * s_theta * s_psi))
    )
    dx[11] = (
        u * (-s_theta)
        + v * (s_phi * c_theta)
        + w * (c_phi * c_theta)
    )

    return dx
