"""Shared feature and label definitions for the clean residual network.

The clean pipeline learns the exact difference between the plant
(``m350_model_real.json``) and the nominal simulator model (``m350_model.json``)
evaluated at the same state. Both the offline data generator (``batch_sim.py``)
and the online predictor (``nn_residual.py``) build their inputs through this
module, so training and inference always see bit-identical features.

Inputs avoid the commanded wrench on purpose: ``u_virtual`` carries the
derivative kick of the 1000 Hz rate loop, while the wrench rebuilt from rotor
speeds is already low-passed by the second-order motor dynamics.
"""

from __future__ import annotations

import numpy as np

from dynamic.quadcopter_model import six_dof_model

BASE_FEATURE_COLS: list[str] = [
    'Vel_u_ms',
    'Vel_v_ms',
    'Vel_w_ms',
    'Rate_p_rads',
    'Rate_q_rads',
    'Rate_r_rads',
    'Rotor_Fz_N',
    'Rotor_Tx_Nm',
    'Rotor_Ty_Nm',
    'Rotor_Tz_Nm',
]

WIND_FEATURE_COLS: list[str] = ['Wind_u_ms', 'Wind_v_ms', 'Wind_w_ms']

FEATURE_SETS: dict[str, list[str]] = {
    'base': BASE_FEATURE_COLS,
    'wind': BASE_FEATURE_COLS + WIND_FEATURE_COLS,
}

LABEL_COLS: list[str] = [
    'Delta_a_u',
    'Delta_a_v',
    'Delta_a_w',
    'Delta_alpha_p',
    'Delta_alpha_q',
    'Delta_alpha_r',
]


def rotor_params_from_dynamics(ac_params: np.ndarray) -> np.ndarray:
    """Extract ``[k_t0, k_q0, dx_arm, dy_fw, dy_bw]`` from a 14-element dynamics array."""
    ac = np.asarray(ac_params, dtype=np.float64)
    return np.array([ac[6], ac[7], ac[11], ac[12], ac[13]], dtype=np.float64)


def rotor_wrench(state: np.ndarray, rotor_params: np.ndarray) -> np.ndarray:
    """Rebuild the rotor wrench from the current rotor speeds.

    Args:
        state: 20-element quadcopter state vector.
        rotor_params: ``[k_t0, k_q0, dx_arm, dy_fw, dy_bw]`` of the nominal model.

    Returns:
        ``[Fz, Tx, Ty, Tz]`` with total thrust ``Fz`` [N] (positive up, along ``-z_b``)
        and torques [N·m] (body frame).
    """
    k_t0, k_q0, dx_arm, dy_fw, dy_bw = rotor_params
    w_sq = np.asarray(state[12:20:2], dtype=np.float64) ** 2
    t1, t2, t3, t4 = k_t0 * w_sq
    q1, q2, q3, q4 = k_q0 * w_sq
    return np.array(
        [
            t1 + t2 + t3 + t4,
            (t1 - t2) * dy_fw + (t4 - t3) * dy_bw,
            (t1 + t2 - t3 - t4) * dx_arm,
            -q1 + q2 - q3 + q4,
        ],
        dtype=np.float64,
    )


def wind_to_body(state: np.ndarray, wind_ned: np.ndarray) -> np.ndarray:
    """Rotate the inertial wind vector into the body frame.

    Args:
        state: 20-element quadcopter state vector (Euler angles at ``[6:9]`` [rad]).
        wind_ned: Wind velocity ``[wx, wy, wz]`` [m/s] (NED frame).

    Returns:
        Wind velocity [m/s] (body frame), same convention as ``six_dof_model``.
    """
    phi, theta, psi = state[6], state[7], state[8]
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_th, c_th = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    wx, wy, wz = wind_ned[0], wind_ned[1], wind_ned[2]
    return np.array(
        [
            wx * c_th * c_psi + wy * c_th * s_psi - wz * s_th,
            wx * (-c_phi * s_psi + s_phi * s_th * c_psi)
            + wy * (c_phi * c_psi + s_phi * s_th * s_psi)
            + wz * s_phi * c_th,
            wx * (s_phi * s_psi + c_phi * s_th * c_psi)
            + wy * (-s_phi * c_psi + c_phi * s_th * s_psi)
            + wz * c_phi * c_th,
        ],
        dtype=np.float64,
    )


def build_features(
    state: np.ndarray,
    wind_ned: np.ndarray,
    rotor_params: np.ndarray,
) -> np.ndarray:
    """Assemble the full clean feature row (base + wind columns).

    Returns:
        13-element vector ordered as ``FEATURE_SETS['wind']``. Slice the first
        10 entries for the ``'base'`` feature set.
    """
    return np.concatenate(
        [
            np.asarray(state[0:6], dtype=np.float64),
            rotor_wrench(state, rotor_params),
            wind_to_body(state, wind_ned),
        ]
    )


def analytic_residual(
    state: np.ndarray,
    plant_params: np.ndarray,
    nominal_params: np.ndarray,
    wind_ned: np.ndarray,
    w_cmds: np.ndarray,
) -> np.ndarray:
    """Compute the exact plant-minus-nominal 6-DoF acceleration residual.

    Both models are evaluated at the same state, wind and rotor commands, so
    the label has no differentiation, filtering, or multi-rate aliasing in it.

    Returns:
        ``[Delta_a_u, Delta_a_v, Delta_a_w]`` [m/s²] and
        ``[Delta_alpha_p, Delta_alpha_q, Delta_alpha_r]`` [rad/s²] (body frame).
    """
    dx_plant = six_dof_model(state, plant_params, wind_ned, w_cmds)
    dx_nom = six_dof_model(state, nominal_params, wind_ned, w_cmds)
    return dx_plant[0:6] - dx_nom[0:6]
