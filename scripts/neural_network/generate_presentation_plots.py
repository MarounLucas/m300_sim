"""Run 5 short benchmark missions (with and without NN) and generate comparison plots.

Executes 5 distinct short scenarios (10 closed-loop flights in total: 5 with NN OFF
and 5 with NN ON on the real M350 plant), exports the resulting CSV logs to
``data/flight_logs/eval_5missions/``, and saves clean comparison figures in
``data/nn_validation/presentation/``:
- ``m01``..``m05`` individual mission plots (3D trajectory, tracking error,
  attitude/rates, control torques, and NN residual injection).
- ``00_summary_5missions_off_vs_on.png`` bar chart comparing all 5 missions.
- ``compare_custom_csvs.png`` when called with ``--csv <off.csv> <on.csv>``.

Usage::

    python3 scripts/neural_network/generate_presentation_plots.py
    python3 scripts/neural_network/generate_presentation_plots.py --csv off.csv on.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_m300')

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

_CURRENT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from control.control import CascadeController  # noqa: E402
from dynamic.mixer import Mixer  # noqa: E402
from dynamic.numerical_integration import forward_euler  # noqa: E402
from dynamic.quadcopter_model import six_dof_model  # noqa: E402
from dynamic.wind_manager import WindManager  # noqa: E402
from neural_network.flight_log_process.batch_sim import (  # noqa: E402
    STATE_IDLE,
    STATE_LANDING,
    STATE_STABILIZING,
    STATE_TAKEOFF,
    STATE_TRACKING,
    _parse_wind_type,
    _parse_yaw_mode,
    load_aircraft_config,
)
from neural_network.nn_residual import ResidualNNPredictor  # noqa: E402
import paths  # noqa: E402
from trajectory.trajectory import PathManager, PolynomialTrajectory  # noqa: E402

DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi

BENCHMARK_MISSIONS: list[dict[str, Any]] = [
    {
        'id': 'm01_nowind_calm',
        'title': 'Missão 1 — Quadrado Rápido Sem Vento (Calmo)',
        'short': 'M1: Sem Vento',
        'wind': {
            'type': 'None',
            'magnitude': 0.0,
            'heading': 0.0,
            'elevation': 0.0,
            'gust': 0.0,
        },
        'trajectory': {
            'speed_xy': 8.0,
            'speed_z': 3.5,
            'yaw_mode': 'Forward (Tangent)',
        },
        'waypoints': [
            {'x': 0.0, 'y': 0.0, 'z': -10.0},
            {'x': 20.0, 'y': 0.0, 'z': -12.0},
            {'x': 20.0, 'y': 20.0, 'z': -12.0},
            {'x': 0.0, 'y': 20.0, 'z': -10.0},
        ],
    },
    {
        'id': 'm02_const_wind',
        'title': 'Missão 2 — Vento Constante Forte (8 m/s Lateral/Frontal)',
        'short': 'M2: Vento Const. 8m/s',
        'wind': {
            'type': 'Constant',
            'magnitude': 8.0,
            'heading': 35.0,
            'elevation': -5.0,
            'gust': 0.0,
        },
        'trajectory': {'speed_xy': 8.5, 'speed_z': 3.5, 'yaw_mode': 'Forward (Tangent)'},
        'waypoints': [
            {'x': 0.0, 'y': 0.0, 'z': -12.0},
            {'x': 25.0, 'y': -10.0, 'z': -15.0},
            {'x': 25.0, 'y': 15.0, 'z': -12.0},
        ],
    },
    {
        'id': 'm03_dryden_gust',
        'title': 'Missão 3 — Turbulência de Dryden (7 m/s + Rajadas 10.5 m/s)',
        'short': 'M3: Rajada Dryden',
        'wind': {
            'type': 'Dryden Gust',
            'magnitude': 7.0,
            'heading': -40.0,
            'elevation': 8.0,
            'gust': 10.5,
        },
        'trajectory': {'speed_xy': 9.0, 'speed_z': 4.0, 'yaw_mode': 'Forward (Tangent)'},
        'waypoints': [
            {'x': 0.0, 'y': 0.0, 'z': -10.0},
            {'x': 22.0, 'y': 12.0, 'z': -16.0},
            {'x': -5.0, 'y': 22.0, 'z': -12.0},
        ],
    },
    {
        'id': 'm04_fast_sprint',
        'title': 'Missão 4 — Sprint Rápido 3D Sem Vento (12 m/s Horizontal, 4.5 m/s Vertical)',
        'short': 'M4: Sprint 12m/s',
        'wind': {
            'type': 'None',
            'magnitude': 0.0,
            'heading': 0.0,
            'elevation': 0.0,
            'gust': 0.0,
        },
        'trajectory': {
            'speed_xy': 12.0,
            'speed_z': 4.5,
            'yaw_mode': 'Forward (Tangent)',
        },
        'waypoints': [
            {'x': 0.0, 'y': 0.0, 'z': -12.0},
            {'x': 35.0, 'y': 0.0, 'z': -22.0},
            {'x': 15.0, 'y': -25.0, 'z': -10.0},
        ],
    },
    {
        'id': 'm05_fast_wind_zigzag',
        'title': 'Missão 5 — Zig-Zag Rápido (11 m/s) + Vento Senoidal/Dryden',
        'short': 'M5: Rápida + Vento',
        'wind': {
            'type': 'Dryden Low-Pass',
            'magnitude': 6.5,
            'heading': 60.0,
            'elevation': -6.0,
            'gust': 9.0,
        },
        'trajectory': {'speed_xy': 11.0, 'speed_z': 4.0, 'yaw_mode': 'Target Point'},
        'waypoints': [
            {'x': 0.0, 'y': 0.0, 'z': -10.0},
            {'x': 20.0, 'y': -18.0, 'z': -18.0},
            {'x': 35.0, 'y': 15.0, 'z': -14.0},
        ],
    },
]

CSV_COLS: list[str] = [
    'Time_s',
    'Pos_N_m',
    'Pos_E_m',
    'Pos_Alt_m',
    'Ref_N_m',
    'Ref_E_m',
    'Ref_Alt_m',
    'Roll_deg',
    'Pitch_deg',
    'Yaw_deg',
    'Vel_u_ms',
    'Vel_v_ms',
    'Vel_w_ms',
    'Rate_p_rads',
    'Rate_q_rads',
    'Rate_r_rads',
    'Accel_x_ms2',
    'Accel_y_ms2',
    'Accel_z_ms2',
    'Alpha_p_rads2',
    'Alpha_q_rads2',
    'Alpha_r_rads2',
    'Thrust_Fz_N',
    'Torque_Tx_Nm',
    'Torque_Ty_Nm',
    'Torque_Tz_Nm',
    'NN_du',
    'NN_dv',
    'NN_dw',
    'NN_dp',
    'NN_dq',
    'NN_dr',
]


def simulate_spec(
    ms_spec: dict[str, Any],
    use_nn: bool,
    out_csv: Path | None = None,
    max_sim_time: float = 90.0,
) -> dict[str, np.ndarray]:
    """Run one benchmark mission on the real plant with NN enabled or disabled."""
    sim_cfg, _ = load_aircraft_config('sim')
    real_cfg, _ = load_aircraft_config('real')

    nn_pred: ResidualNNPredictor | None = None
    if use_nn:
        wind_dir = paths.NN_MODELS_DIR / 'wind'
        if not (wind_dir / 'best_m350_model.npz').exists():
            wind_dir = paths.NN_MODELS_DIR
        nn_pred = ResidualNNPredictor(model_dir=wind_dir, enabled=True)

    wps = [[float(w['x']), float(w['y']), float(w['z'])] for w in ms_spec['waypoints']]
    traj_cfg = ms_spec.get('trajectory', {})
    yaw_mode = _parse_yaw_mode(traj_cfg.get('yaw_mode', 'Forward (Tangent)'))
    speed_xy = float(traj_cfg.get('speed_xy', 9.0))
    speed_z = float(traj_cfg.get('speed_z', 3.5))

    path_manager = PathManager(
        wps,
        yaw_mode=yaw_mode,
        yaw_target=[15.0, 0.0],
        xy_velocity=speed_xy,
        z_velocity=speed_z,
    )
    for n in range(len(path_manager.waypoints) - 1):
        q0 = path_manager.waypoints[n]
        qf = path_manager.waypoints[n + 1]
        t_seg = path_manager.calculate_segment_time(q0, qf)
        path_manager.start_times.append(path_manager.total_time)
        path_manager.segment_times.append(t_seg)
        path_manager.total_time += t_seg
        seg_traj = PolynomialTrajectory(q0[0:3], qf[0:3], t_seg)
        path_manager.segments.append(seg_traj)
        path_manager.segment_coefficients.append(seg_traj.generate_path())

    wind_cfg = ms_spec.get('wind', {})
    # Fixed seed per mission so NN OFF and NN ON experience the exact same gust sequence
    np.random.seed(12345)
    wind_manager = WindManager(
        wind_type=_parse_wind_type(wind_cfg.get('type', 'None')),
        magnitude=float(wind_cfg.get('magnitude', 0.0)),
        heading=float(wind_cfg.get('heading', 0.0)) * DEG_TO_RAD,
        elevation=float(wind_cfg.get('elevation', 0.0)) * DEG_TO_RAD,
        gust_magnitude=float(wind_cfg.get('gust', 0.0)),
    )

    ac_params = real_cfg.to_dynamics_array()
    mixer = Mixer(sim_cfg)
    controller = CascadeController(sim_cfg)

    omega_hover = math.sqrt((real_cfg.mass * 9.81) / (4.0 * real_cfg.k_t0))
    state = np.zeros(20, dtype=np.float64)
    state[12:20:2] = omega_hover

    hover_thrust = sim_cfg.mass * 9.81
    u_virtual = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float64)
    total_thrust = hover_thrust
    curr_wind = np.zeros(3, dtype=np.float64)
    curr_nn = np.zeros(6, dtype=np.float64)

    curr_position = np.zeros(3, dtype=np.float64)
    target_wp0 = path_manager.waypoints[0][0:3]
    takeoff_dist = float(np.linalg.norm(curr_position - target_wp0))
    recovery_duration = max(3.0, takeoff_dist / 2.5)
    recovery_traj = PolynomialTrajectory(curr_position, target_wp0, recovery_duration)
    recovery_traj.generate_path()
    recovery_time = 0.0

    mission_state = STATE_TAKEOFF
    traj_time = 0.0
    segment_idx = 0
    stabilize_time = 0.0
    pos_ref = np.zeros(3, dtype=np.float64)
    vel_ref = np.zeros(3, dtype=np.float64)
    yaw_ref = 0.0

    dt = 0.001
    dt_traj = 0.01
    max_ticks = int(max_sim_time / dt)
    ctrl_state = np.zeros(12, dtype=np.float64)
    waypoint_cmd = np.zeros(4, dtype=np.float64)
    rows: list[list[float]] = []

    for tick in range(max_ticks):
        t_sim = tick * dt
        if tick % 10 == 0:
            curr_wind = np.asarray(
                wind_manager.get_wind(t_sim, state[9:12]), dtype=np.float64
            )
            if nn_pred is not None:
                curr_nn = nn_pred.step(state, u_virtual, curr_wind)

        w_cmds = mixer.compute_motor_speed(u_virtual)
        # Subtract nn_residual so the NN cancels the (real - nominal) plant mismatch
        dx = six_dof_model(
            state,
            ac_params,
            curr_wind,
            w_cmds,
            -curr_nn if nn_pred is not None else None,
        )
        state = forward_euler(state, dx, dt)

        rot = Rotation.from_euler('xyz', state[6:9])
        roll, pitch, yaw = rot.as_euler('xyz')

        ctrl_state[0:6] = state[0:6]
        ctrl_state[6:9] = roll, pitch, yaw
        ctrl_state[9:12] = state[9:12]
        controller.update_state(ctrl_state)
        curr_position[0:3] = state[9:12]

        if (tick + 1) % 10 == 0:
            if mission_state == STATE_TAKEOFF:
                recovery_time += dt_traj
                pos_ref = recovery_traj.sample_position(recovery_time)
                vel_ref = recovery_traj.sample_velocity(recovery_time)
                yaw_ref = (
                    path_manager.calculate_yaw(0, 0.0, pos_ref)
                    if path_manager.yaw_mode == 'target'
                    else 0.0
                )
                if recovery_time >= recovery_duration:
                    traj_time = 0.0
                    segment_idx = 0
                    mission_state = STATE_TRACKING
            elif mission_state == STATE_TRACKING:
                traj_time += dt_traj
                start_t = path_manager.start_times[segment_idx]
                seg_dur = path_manager.segment_times[segment_idx]
                local_time = traj_time - start_t
                sample_t = min(traj_time, start_t + seg_dur - 1e-6)
                pos_ref, vel_ref, yaw_ref, _ = path_manager.get_desired_state(sample_t)
                if local_time >= seg_dur:
                    mission_state = STATE_STABILIZING
                    stabilize_time = 0.0
            elif mission_state == STATE_STABILIZING:
                target_pos = path_manager.waypoints[segment_idx + 1][0:3].copy()
                pos_ref = target_pos
                vel_ref = np.zeros(3)
                yaw_ref = (
                    path_manager.calculate_yaw(segment_idx, 0.0, target_pos)
                    if path_manager.yaw_mode == 'target'
                    else path_manager.last_yaw
                )
                if float(np.linalg.norm(curr_position - target_pos)) < 0.20:
                    stabilize_time += dt_traj
                    if stabilize_time >= 1.5:
                        if segment_idx < (len(path_manager.segments) - 1):
                            segment_idx += 1
                            traj_time = path_manager.start_times[segment_idx]
                            mission_state = STATE_TRACKING
                        else:
                            mission_state = STATE_IDLE
                else:
                    stabilize_time = 0.0
            elif mission_state == STATE_LANDING:
                mission_state = STATE_IDLE

            waypoint_cmd[0:3] = pos_ref
            waypoint_cmd[3] = yaw_ref
            controller.desired_state(waypoint_cmd, vel_ref)

        if tick % 20 == 0:
            controller.update_auto_guidance()
            controller._xy_vel_control(dt=0.02)
            controller._z_vel_control(dt=0.02)
            total_thrust = controller._accel_to_attitude()

        if tick % 4 == 0:
            controller._angle_control()

        target_torque = controller._angular_rate_control(dt=dt)
        u_virtual[0] = total_thrust
        u_virtual[1:4] = target_torque

        if tick % 10 == 0:
            rows.append([
                t_sim,
                float(state[9]),
                float(state[10]),
                float(-state[11]),
                float(pos_ref[0]),
                float(pos_ref[1]),
                float(-pos_ref[2]),
                float(roll * RAD_TO_DEG),
                float(pitch * RAD_TO_DEG),
                float(yaw * RAD_TO_DEG),
                float(state[0]),
                float(state[1]),
                float(state[2]),
                float(state[3]),
                float(state[4]),
                float(state[5]),
                float(dx[0]),
                float(dx[1]),
                float(dx[2]),
                float(dx[3]),
                float(dx[4]),
                float(dx[5]),
                float(u_virtual[0]),
                float(u_virtual[1]),
                float(u_virtual[2]),
                float(u_virtual[3]),
                *curr_nn.tolist(),
            ])

        if mission_state == STATE_IDLE:
            break

    mat = np.asarray(rows, dtype=np.float64)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_csv,
            mat,
            delimiter=',',
            header=','.join(CSV_COLS),
            comments='',
            fmt='%.6f',
        )
    return {c: mat[:, i] for i, c in enumerate(CSV_COLS)}


def load_csv_log(filepath: Path) -> dict[str, np.ndarray]:
    """Read a flight log CSV into a dictionary of float64 NumPy arrays."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader if row]
    mat = np.asarray(rows, dtype=np.float64)
    out = {col: mat[:, i] for i, col in enumerate(header)}
    out['Time_s'] = mat[:, 0] - mat[0, 0]
    return out


def compute_pair_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute tracking error, attitude smoothness, and torque effort metrics."""
    t = log['Time_s']
    alt = log.get('Pos_Alt_m', np.zeros_like(t))
    air = alt > 0.3
    if int(np.sum(air)) < 10:
        air = np.ones_like(t, dtype=bool)

    metrics: dict[str, float] = {
        'duration_s': float(t[-1]),
        'sigma_p': float(np.std(log['Rate_p_rads'][air])),
        'sigma_q': float(np.std(log['Rate_q_rads'][air])),
        'sigma_alpha_p': float(np.std(log['Alpha_p_rads2'][air])),
        'sigma_alpha_q': float(np.std(log['Alpha_q_rads2'][air])),
        'sigma_tx': float(np.std(log['Torque_Tx_Nm'][air])),
        'sigma_ty': float(np.std(log['Torque_Ty_Nm'][air])),
        'mean_fz': float(np.mean(log['Thrust_Fz_N'][air])),
    }
    if 'Ref_N_m' in log:
        err_xy = np.hypot(
            log['Pos_N_m'][air] - log['Ref_N_m'][air],
            log['Pos_E_m'][air] - log['Ref_E_m'][air],
        )
        err_z = np.abs(log['Pos_Alt_m'][air] - log['Ref_Alt_m'][air])
        err_3d = np.hypot(err_xy, err_z)
        metrics['rmse_3d_m'] = float(np.sqrt(np.mean(err_3d**2)))
        metrics['rmse_z_m'] = float(np.sqrt(np.mean(err_z**2)))
        metrics['max_err_3d_m'] = float(np.max(err_3d))
    return metrics


def plot_single_mission_pair(
    ms_spec: dict[str, Any],
    log_off: dict[str, np.ndarray],
    log_on: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    """Plot a 6-panel comparison figure for one mission (NN OFF vs. NN ON)."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    t_off = log_off['Time_s']
    t_on = log_on['Time_s']

    # 1. Top-Down XY Path
    ax = axs[0, 0]
    ax.plot(
        log_off['Ref_E_m'],
        log_off['Ref_N_m'],
        'k:',
        lw=1.8,
        label='Referência Desejada',
    )
    ax.plot(
        log_off['Pos_E_m'],
        log_off['Pos_N_m'],
        color='#d62728',
        linestyle='--',
        lw=1.6,
        label='Sem NN (NN OFF)',
    )
    ax.plot(
        log_on['Pos_E_m'],
        log_on['Pos_N_m'],
        color='#2ca02c',
        lw=2.0,
        label='Com NN (NN ON)',
    )
    ax.set_title('Trajetória Horizontal (Norte vs. Leste)', fontweight='bold')
    ax.set_xlabel('Leste [m]')
    ax.set_ylabel('Norte [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    # 2. Altitude Tracking
    ax = axs[0, 1]
    ax.plot(t_off, log_off['Ref_Alt_m'], 'k:', lw=1.6, label='Altitude Ref.')
    ax.plot(t_off, log_off['Pos_Alt_m'], color='#d62728', ls='--', lw=1.6, label='Sem NN')
    ax.plot(t_on, log_on['Pos_Alt_m'], color='#2ca02c', lw=2.0, label='Com NN')
    ax.set_title('Rastreamento de Altitude [m]', fontweight='bold')
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('Altitude [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    # 3. Instantaneous 3D Position Error
    ax = axs[1, 0]
    err_off = np.sqrt(
        (log_off['Pos_N_m'] - log_off['Ref_N_m']) ** 2
        + (log_off['Pos_E_m'] - log_off['Ref_E_m']) ** 2
        + (log_off['Pos_Alt_m'] - log_off['Ref_Alt_m']) ** 2
    )
    err_on = np.sqrt(
        (log_on['Pos_N_m'] - log_on['Ref_N_m']) ** 2
        + (log_on['Pos_E_m'] - log_on['Ref_E_m']) ** 2
        + (log_on['Pos_Alt_m'] - log_on['Ref_Alt_m']) ** 2
    )
    ax.plot(t_off, err_off, color='#d62728', ls='--', lw=1.6, label='Erro 3D — Sem NN')
    ax.plot(t_on, err_on, color='#2ca02c', lw=2.0, label='Erro 3D — Com NN')
    ax.set_title('Erro Instantâneo de Posição 3D [m]', fontweight='bold')
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('||p - p_ref|| [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    # 4. Control Torques Tx & Ty
    ax = axs[1, 1]
    ax.plot(t_off, log_off['Torque_Tx_Nm'], color='#d62728', ls='--', lw=1.4, label='Tx Sem NN')
    ax.plot(t_on, log_on['Torque_Tx_Nm'], color='#2ca02c', lw=1.8, label='Tx Com NN')
    ax.plot(t_off, log_off['Torque_Ty_Nm'], color='#ff7f0e', ls='--', lw=1.4, label='Ty Sem NN')
    ax.plot(t_on, log_on['Torque_Ty_Nm'], color='#1f77b4', lw=1.8, label='Ty Com NN')
    ax.set_title('Torques de Controle (Roll Tx e Pitch Ty) [N·m]', fontweight='bold')
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('Torque [N·m]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # 5. Angular Accelerations alpha_p & alpha_q
    ax = axs[2, 0]
    ax.plot(t_off, log_off['Alpha_p_rads2'], color='#d62728', ls='--', lw=1.4, label='αp Sem NN')
    ax.plot(t_on, log_on['Alpha_p_rads2'], color='#2ca02c', lw=1.8, label='αp Com NN')
    ax.plot(t_off, log_off['Alpha_q_rads2'], color='#ff7f0e', ls='--', lw=1.4, label='αq Sem NN')
    ax.plot(t_on, log_on['Alpha_q_rads2'], color='#1f77b4', lw=1.8, label='αq Com NN')
    ax.set_title('Acelerações Angulares (Roll αp e Pitch αq) [rad/s²]', fontweight='bold')
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('[rad/s²]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # 6. NN Predicted Residuals (Active during NN ON)
    ax = axs[2, 1]
    ax.plot(t_on, log_on['NN_du'], lw=1.6, label='Δau [m/s²]')
    ax.plot(t_on, log_on['NN_dv'], lw=1.6, label='Δav [m/s²]')
    ax.plot(t_on, log_on['NN_dw'], lw=1.8, label='Δaw [m/s²]')
    ax.plot(t_on, log_on['NN_dr'], lw=1.6, label='Δαr [rad/s²]')
    ax.set_title('Compensação Residual Predita pela NN (Voo Limpo)', fontweight='bold')
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('Resíduo Predito')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(ms_spec['title'], fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = out_dir / f"{ms_spec['id']}_off_vs_on.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_summary_5missions(
    summary_rows: list[tuple[str, dict[str, float], dict[str, float]]],
    out_dir: Path,
) -> Path:
    """Generate a 4-panel bar chart summarizing all 5 missions (NN OFF vs. NN ON)."""
    labels = [r[0] for r in summary_rows]
    x = np.arange(len(labels))
    w = 0.35

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    panels = [
        ('rmse_3d_m', 'Erro RMSE de Posição 3D [m] (Menor = Melhor)', axs[0, 0]),
        ('rmse_z_m', 'Erro RMSE de Altitude [m] (Menor = Melhor)', axs[0, 1]),
        ('sigma_tx', 'Desvio Padrão Torque Roll σ(Tx) [N·m]', axs[1, 0]),
        ('sigma_alpha_q', 'Desvio Padrão Acel. Pitch σ(αq) [rad/s²]', axs[1, 1]),
    ]

    for key, title, ax in panels:
        v_off = [r[1].get(key, 0.0) for r in summary_rows]
        v_on = [r[2].get(key, 0.0) for r in summary_rows]
        b1 = ax.bar(x - w / 2, v_off, width=w, color='#d62728', label='Sem NN (NN OFF)')
        b2 = ax.bar(x + w / 2, v_on, width=w, color='#2ca02c', label='Com NN (NN ON)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=9)
        for bar in (*b1, *b2):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.01,
                f'{h:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    fig.suptitle(
        'Resumo Comparativo das 5 Missões (10 Voos: 5 Sem NN vs. 5 Com NN Limpa)',
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout()
    out_path = out_dir / '00_summary_5missions_off_vs_on.png'
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def compare_custom_csvs(csv_paths: list[Path], out_dir: Path) -> Path:
    """Plot a direct visual comparison of arbitrary CSV flight logs passed via ``--csv``."""
    logs = [(p.stem, load_csv_log(p)) for p in csv_paths if p.exists()]
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    specs = [
        ('Pos_Alt_m', 'Altitude [m]'),
        ('Thrust_Fz_N', 'Empuxo Total Fz [N]'),
        ('Torque_Tx_Nm', 'Torque de Roll Tx [N·m]'),
        ('Torque_Ty_Nm', 'Torque de Pitch Ty [N·m]'),
        ('Alpha_p_rads2', 'Aceleração Angular Roll αp [rad/s²]'),
        ('Alpha_q_rads2', 'Aceleração Angular Pitch αq [rad/s²]'),
    ]
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
    for idx, (col, title) in enumerate(specs):
        r, c = divmod(idx, 2)
        ax = axs[r, c]
        for i, (name, log) in enumerate(logs):
            if col in log:
                ax.plot(
                    log['Time_s'],
                    log[col],
                    color=colors[i % len(colors)],
                    lw=1.6,
                    label=name,
                )
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Tempo [s]')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8.5)

    fig.suptitle('Comparativo de Logs CSV', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = out_dir / 'compare_custom_csvs.png'
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Run the 5 benchmark missions (10 flights) or compare custom CSV files."""
    parser = argparse.ArgumentParser(
        description='Run 5 short missions (with and without NN) and generate plots.'
    )
    parser.add_argument(
        '--csv',
        nargs='+',
        help='Optional list of custom CSV files to compare directly.',
    )
    args = parser.parse_args(argv)

    out_dir = paths.NN_VALIDATION_DIR / 'presentation'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        out_p = compare_custom_csvs([Path(p) for p in args.csv], out_dir)
        print(f'Saved custom CSV comparison: {out_p.relative_to(paths.PROJECT_ROOT)}')
        return 0

    csv_dir = paths.FLIGHT_LOGS_DIR / 'eval_5missions'
    csv_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[str, dict[str, float], dict[str, float]]] = []
    generated: list[Path] = []

    print('\n=== Executando 5 Missões Curtas (10 Voos: 5 Sem NN vs. 5 Com NN) ===')
    print(
        f"{'Missão':<24} | {'RMSE 3D Off':>11} {'RMSE 3D On':>11} | "
        f"{'RMSE Z Off':>10} {'RMSE Z On':>10} | {'σ(Ty) Off':>10} {'σ(Ty) On':>10}"
    )
    print('-' * 98)

    for ms in BENCHMARK_MISSIONS:
        mid = ms['id']
        csv_off = csv_dir / f'{mid}_nn_off.csv'
        csv_on = csv_dir / f'{mid}_nn_on.csv'

        log_off = simulate_spec(ms, use_nn=False, out_csv=csv_off)
        log_on = simulate_spec(ms, use_nn=True, out_csv=csv_on)

        m_off = compute_pair_metrics(log_off)
        m_on = compute_pair_metrics(log_on)
        summary_rows.append((ms['short'], m_off, m_on))

        print(
            f"{ms['short']:<24} | {m_off['rmse_3d_m']:>10.3f}m {m_on['rmse_3d_m']:>10.3f}m | "
            f"{m_off['rmse_z_m']:>9.3f}m {m_on['rmse_z_m']:>9.3f}m | "
            f"{m_off['sigma_ty']:>10.4f} {m_on['sigma_ty']:>10.4f}"
        )

        fig_p = plot_single_mission_pair(ms, log_off, log_on, out_dir)
        generated.append(fig_p)

    summary_fig = plot_summary_5missions(summary_rows, out_dir)
    generated.insert(0, summary_fig)

    for p in generated:
        try:
            os.chmod(p, 0o666)
        except OSError:
            pass

    print('\nCSVs salvos em:', csv_dir.relative_to(paths.PROJECT_ROOT))
    print('Imagens salvas em:')
    for p in generated:
        print(f'  - {p.relative_to(paths.PROJECT_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
