"""Headless batch flight simulator for generating neural network training logs.

Executes the 6-DoF quadcopter physics, multi-rate PX4 cascade controller,
polynomial trajectory planner, and stochastic wind models in an accelerated
offline loop across all missions in ``data/offline_missions`` (``training``,
``validation``, and ``test``) for any chosen aircraft model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

_CURRENT_DIR = Path(__file__).resolve().parent
_NN_DIR = _CURRENT_DIR.parent
_SCRIPTS_DIR = _NN_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _install_rotorpy_fallback() -> None:
    """Register fallback RotorPy wind classes when running outside the ROS 2 container."""
    try:
        import rotorpy.wind.default_winds  # noqa: F401
        import rotorpy.wind.dryden_winds  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _NoWind:
        def update(self, t: float, position: np.ndarray) -> np.ndarray:
            return np.zeros(3, dtype=np.float64)

    class _ConstantWind:
        def __init__(self, wx: float = 0.0, wy: float = 0.0, wz: float = 0.0) -> None:
            self._vec = np.array([wx, wy, wz], dtype=np.float64)

        def update(self, t: float, position: np.ndarray) -> np.ndarray:
            return self._vec.copy()

    class _SinusoidWind:
        def __init__(
            self, amplitudes: np.ndarray, frequencies: np.ndarray
        ) -> None:
            self._amp = np.asarray(amplitudes, dtype=np.float64)
            self._freq = np.asarray(frequencies, dtype=np.float64)

        def update(self, t: float, position: np.ndarray) -> np.ndarray:
            return self._amp * np.sin(2.0 * math.pi * self._freq * t)

    class _DrydenGust:
        def __init__(
            self,
            avg_wind: np.ndarray,
            sig_wind: np.ndarray,
            altitude: float = 10.0,
            dt: float = 0.01,
        ) -> None:
            self.avg_wind = np.asarray(avg_wind, dtype=np.float64)
            self.sig_wind = np.asarray(sig_wind, dtype=np.float64)
            self.altitude = max(1.0, float(altitude))
            self.dt = dt
            self.state = np.zeros(3, dtype=np.float64)
            self._last_t: float | None = None

        def update(self, t: float, position: np.ndarray) -> np.ndarray:
            dt = (
                self.dt
                if self._last_t is None or t <= self._last_t
                else min(0.1, t - self._last_t)
            )
            self._last_t = t
            v_mean = max(1.0, float(np.linalg.norm(self.avg_wind)))
            scale = np.array(
                [
                    self.altitude / ((0.177 + 0.000823 * self.altitude) ** 1.2),
                    self.altitude / ((0.177 + 0.000823 * self.altitude) ** 1.2),
                    self.altitude,
                ],
                dtype=np.float64,
            )
            tau_d = np.maximum(0.05, scale / v_mean)
            alpha = np.exp(-dt / tau_d)
            noise = np.random.normal(0.0, 1.0, size=3)
            self.state = (
                alpha * self.state
                + np.sqrt(1.0 - alpha**2) * self.sig_wind * noise
            )
            return self.avg_wind + self.state

    class _DrydenGustLP(_DrydenGust):
        def __init__(
            self,
            avg_wind: np.ndarray,
            sig_wind: np.ndarray,
            altitude: float = 10.0,
            tau: float = 0.5,
        ) -> None:
            super().__init__(avg_wind=avg_wind, sig_wind=sig_wind, altitude=altitude)
            self.tau = max(1e-3, float(tau))
            self.lp_state = self.avg_wind.copy()

        def update(self, t: float, position: np.ndarray) -> np.ndarray:
            raw = super().update(t, position)
            alpha_lp = math.exp(-self.dt / self.tau)
            self.lp_state = alpha_lp * self.lp_state + (1.0 - alpha_lp) * raw
            return self.lp_state.copy()

    rotorpy_mod = types.ModuleType('rotorpy')
    wind_mod = types.ModuleType('rotorpy.wind')
    default_mod = types.ModuleType('rotorpy.wind.default_winds')
    dryden_mod = types.ModuleType('rotorpy.wind.dryden_winds')

    default_mod.NoWind = _NoWind  # type: ignore[attr-defined]
    default_mod.ConstantWind = _ConstantWind  # type: ignore[attr-defined]
    default_mod.SinusoidWind = _SinusoidWind  # type: ignore[attr-defined]
    dryden_mod.DrydenGust = _DrydenGust  # type: ignore[attr-defined]
    dryden_mod.DrydenGustLP = _DrydenGustLP  # type: ignore[attr-defined]

    sys.modules['rotorpy'] = rotorpy_mod
    sys.modules['rotorpy.wind'] = wind_mod
    sys.modules['rotorpy.wind.default_winds'] = default_mod
    sys.modules['rotorpy.wind.dryden_winds'] = dryden_mod


_install_rotorpy_fallback()

from control.control import CascadeController  # noqa: E402
from dynamic.mixer import Mixer  # noqa: E402
from dynamic.numerical_integration import forward_euler  # noqa: E402
from dynamic.quadcopter_model import six_dof_model  # noqa: E402
from dynamic.wind_manager import WindManager  # noqa: E402
from neural_network.nn_features import (  # noqa: E402
    analytic_residual,
    build_features,
    FEATURE_SETS,
    LABEL_COLS,
    rotor_params_from_dynamics,
)
import paths  # noqa: E402
from trajectory.trajectory import PathManager, PolynomialTrajectory  # noqa: E402

DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi

STATE_IDLE = 'IDLE'
STATE_TAKEOFF = 'TAKEOFF'
STATE_TRACKING = 'TRACKING'
STATE_STABILIZING = 'STABILIZING'
STATE_LANDING = 'LANDING'

CSV_HEADER = [
    'Time_s',
    'Pos_N_m',
    'Pos_E_m',
    'Pos_Alt_m',
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
]

CLEAN_CSV_HEADER = [
    'Time_s',
    'Pos_N_m',
    'Pos_E_m',
    'Pos_Alt_m',
    'Roll_deg',
    'Pitch_deg',
    'Yaw_deg',
    *FEATURE_SETS['wind'],
    *LABEL_COLS,
]


@dataclass
class AircraftConfig:
    """Physical, propulsion, and kinematic parameters of an aircraft model.

    Attributes:
        name (str): Human-readable model name.
        mass (float): Total vehicle mass [kg].
        jx (float): Roll moment of inertia [kg·m²] (body frame).
        jy (float): Pitch moment of inertia [kg·m²] (body frame).
        jz (float): Yaw moment of inertia [kg·m²] (body frame).
        jxz (float): XZ product of inertia [kg·m²] (body frame).
        cd (float): Linear aerodynamic drag coefficient [N/(m/s)].
        k_t0 (float): Rotor thrust coefficient [N/(rad/s)²].
        k_q0 (float): Rotor reactive torque coefficient [N·m/(rad/s)²].
        tau (float): Motor second-order time constant [s].
        kp (float): Motor static gain [dimensionless].
        xi (float): Motor damping ratio [dimensionless].
        dx_arm (float): Longitudinal arm distance from center of mass [m].
        dy_fw (float): Front lateral arm distance from center of mass [m].
        dy_bw (float): Rear lateral arm distance from center of mass [m].
        omega_min (float): Minimum rotor speed [rad/s].
        omega_max (float): Maximum rotor speed [rad/s].
        max_ascent_speed (float): Maximum climb velocity [m/s] (NED frame, negative up).
        max_descent_speed (float): Maximum descent velocity [m/s] (NED frame).
        cruise_speed (float): Nominal horizontal cruise speed [m/s].
        max_tilt_angle (float): Maximum roll/pitch tilt angle [rad].
        max_roll_pitch_rate (float): Maximum roll/pitch angular rate [rad/s].
        max_yaw_rate (float): Maximum yaw angular rate [rad/s].
    """

    name: str
    mass: float
    jx: float
    jy: float
    jz: float
    jxz: float
    cd: float
    k_t0: float
    k_q0: float
    tau: float
    kp: float
    xi: float
    dx_arm: float
    dy_fw: float
    dy_bw: float
    omega_min: float
    omega_max: float
    max_ascent_speed: float
    max_descent_speed: float
    cruise_speed: float
    max_tilt_angle: float
    max_roll_pitch_rate: float
    max_yaw_rate: float

    def to_dynamics_array(self) -> np.ndarray:
        """Pack the 14 physical parameters expected by ``six_dof_model``."""
        return np.array(
            [
                self.mass,
                self.jx,
                self.jy,
                self.jz,
                self.jxz,
                self.cd,
                self.k_t0,
                self.k_q0,
                self.tau,
                self.kp,
                self.xi,
                self.dx_arm,
                self.dy_fw,
                self.dy_bw,
            ],
            dtype=np.float64,
        )


def load_aircraft_config(aircraft_arg: str) -> tuple[AircraftConfig, str]:
    """Load an aircraft JSON configuration by name, alias, or file path.

    Args:
        aircraft_arg: Alias (``'real'``, ``'sim'``), model stem
            (``'m350_model_real'``, ``'m350_model'``), or path to a JSON file.

    Returns:
        Tuple ``(config, target_subfolder)`` where ``target_subfolder`` is
        ``'real'``, ``'sim'``, or a sanitized model folder name.
    """
    alias_map = {
        'real': paths.AIRCRAFT_MODELS_DIR / 'm350_model_real.json',
        'm350_model_real': paths.AIRCRAFT_MODELS_DIR / 'm350_model_real.json',
        'sim': paths.AIRCRAFT_MODELS_DIR / 'm350_model.json',
        'm350_model': paths.AIRCRAFT_MODELS_DIR / 'm350_model.json',
    }

    if aircraft_arg in alias_map:
        json_path = alias_map[aircraft_arg]
    else:
        candidate = Path(aircraft_arg)
        if not candidate.exists():
            candidate = paths.AIRCRAFT_MODELS_DIR / (
                aircraft_arg
                if aircraft_arg.endswith('.json')
                else f'{aircraft_arg}.json'
            )
        json_path = candidate

    if not json_path.exists():
        raise FileNotFoundError(f'Aircraft model file not found: {json_path}')

    with open(json_path, 'r', encoding='utf-8') as f:
        data: dict[str, Any] = json.load(f)

    stem = json_path.stem.lower()
    if 'real' in stem:
        target_sub = 'real'
    elif stem in ('m350_model', 'sim'):
        target_sub = 'sim'
    else:
        target_sub = stem

    cfg = AircraftConfig(
        name=str(data.get('name', stem)),
        mass=float(data['mass']),
        jx=float(data['jx']),
        jy=float(data['jy']),
        jz=float(data['jz']),
        jxz=float(data.get('jxz', 0.0)),
        cd=float(data['cd']),
        k_t0=float(data['k_t0']),
        k_q0=float(data['k_q0']),
        tau=float(data['tau']),
        kp=float(data['kp']),
        xi=float(data['xi']),
        dx_arm=float(data['dx_arm']),
        dy_fw=float(data['dy_fw']),
        dy_bw=float(data['dy_bw']),
        omega_min=float(data['omega_min']),
        omega_max=float(data['omega_max']),
        max_ascent_speed=float(data['max_ascent_speed']),
        max_descent_speed=float(data['max_descent_speed']),
        cruise_speed=float(data['cruise_speed']),
        max_tilt_angle=float(data['max_tilt_angle']),
        max_roll_pitch_rate=float(data['max_roll_pitch_rate']),
        max_yaw_rate=float(data['max_yaw_rate']),
    )
    return cfg, target_sub


def _parse_yaw_mode(ui_text: str) -> str:
    """Map GUI yaw mode description to internal ``PathManager`` mode string."""
    if 'Forward' in ui_text:
        return 'forward'
    if 'Target' in ui_text:
        return 'target'
    return 'free'


def _parse_wind_type(ui_text: str) -> str:
    """Map GUI wind model description to internal ``WindManager`` identifier."""
    raw = ui_text.lower()
    if 'constant' in raw:
        return 'constant'
    if 'dryden gust' in raw:
        return 'dryden'
    if 'dryden low' in raw:
        return 'dryden_lp'
    if 'sinusoid' in raw:
        return 'sinusoid'
    return 'none'


def simulate_mission(
    ac_cfg: AircraftConfig,
    mission_path: Path,
    output_csv_path: Path,
    log_hz: float = 100.0,
    max_sim_time: float = 600.0,
    nominal_cfg: AircraftConfig | None = None,
) -> tuple[str, float, int]:
    """Run a single mission offline and write the resulting CSV flight log.

    Replicates the exact multi-rate architecture of the ROS 2 nodes:
    physics integration at 1000 Hz, trajectory FSM and wind updates at 100 Hz,
    attitude control at 250 Hz, and position/velocity guidance at 50 Hz.

    Args:
        ac_cfg: Physical and control parameters of the aircraft.
        mission_path: Path to the mission JSON file.
        output_csv_path: Destination path for the generated CSV log.
        log_hz: Telemetry logging frequency [Hz].
        max_sim_time: Safety timeout in simulated flight time [s].
        nominal_cfg: When given, switch to clean mode and log the rotor-based features
            and the analytic ``ac_cfg`` minus ``nominal_cfg`` residual labels instead of
            the raw telemetry columns.

    Returns:
        Tuple ``(mission_name, simulated_duration_s, num_logged_rows)``.
    """
    # crc32 instead of hash(): str hashes are salted per process, which made the
    # Dryden gust sequence change between runs.
    np.random.seed(zlib.crc32(mission_path.stem.encode('utf-8')))

    with open(mission_path, 'r', encoding='utf-8') as f:
        ms_data = json.load(f)

    wps = [
        [float(w['x']), float(w['y']), float(w['z'])]
        for w in ms_data['waypoints']
    ]
    traj_cfg = ms_data.get('trajectory', {})
    yaw_mode = _parse_yaw_mode(traj_cfg.get('yaw_mode', 'Forward (Tangent)'))
    yaw_t = traj_cfg.get('yaw_target', {})
    yaw_target = [float(yaw_t.get('x', 0.0)), float(yaw_t.get('y', 0.0))]
    speed_xy = float(traj_cfg.get('speed_xy', 9.2))
    speed_z = float(traj_cfg.get('speed_z', 2.0))

    # Suppress stdout from PathManager.generate_path during parallel batch runs
    path_manager = PathManager(
        wps,
        yaw_mode=yaw_mode,
        yaw_target=yaw_target,
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

    wind_cfg = ms_data.get('wind', {})
    wind_manager = WindManager(
        wind_type=_parse_wind_type(wind_cfg.get('type', 'None')),
        magnitude=float(wind_cfg.get('magnitude', 0.0)),
        heading=float(wind_cfg.get('heading', 0.0)) * DEG_TO_RAD,
        elevation=float(wind_cfg.get('elevation', 0.0)) * DEG_TO_RAD,
        gust_magnitude=float(wind_cfg.get('gust', 0.0)),
    )

    ac_params = ac_cfg.to_dynamics_array()
    clean = nominal_cfg is not None
    clean_rows: list[list[float]] = []
    if clean:
        nominal_params = nominal_cfg.to_dynamics_array()
        rotor_params = rotor_params_from_dynamics(nominal_params)
    mixer = Mixer(ac_cfg)
    controller = CascadeController(ac_cfg)

    safe_k_t0 = ac_cfg.k_t0 if ac_cfg.k_t0 > 0.0 else 0.0002433
    omega_hover = math.sqrt((ac_cfg.mass * 9.81) / (4.0 * safe_k_t0))

    state = np.array(
        [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            omega_hover, 0.0,
            omega_hover, 0.0,
            omega_hover, 0.0,
            omega_hover, 0.0,
        ],
        dtype=np.float64,
    )

    hover_thrust = ac_cfg.mass * 9.81
    u_virtual = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float64)
    total_thrust = hover_thrust
    target_torque = np.zeros(3, dtype=np.float64)
    curr_wind = np.zeros(3, dtype=np.float64)

    # Initialize trajectory FSM in STATE_TAKEOFF
    curr_position = np.zeros(3, dtype=np.float64)
    target_wp0 = path_manager.waypoints[0][0:3]
    takeoff_dist = float(np.linalg.norm(curr_position - target_wp0))
    recovery_duration = max(3.0, takeoff_dist / 2.0)
    recovery_traj = PolynomialTrajectory(
        curr_position, target_wp0, recovery_duration
    )
    recovery_traj.generate_path()
    recovery_time = 0.0

    mission_state = STATE_TAKEOFF
    traj_time = 0.0
    segment_idx = 0
    stabilize_time = 0.0
    wp_tolerance = 0.15

    dt = 0.001
    dt_traj = 0.01
    max_ticks = int(max_sim_time / dt)
    log_stride = max(1, int(round(1000.0 / log_hz)))

    rows: list[list[float]] = []
    ctrl_state = np.zeros(12, dtype=np.float64)
    waypoint_cmd = np.zeros(4, dtype=np.float64)

    for tick in range(max_ticks):
        t_sim = tick * dt

        if tick % 10 == 0:
            curr_wind = np.asarray(
                wind_manager.get_wind(t_sim, state[9:12]), dtype=np.float64
            )

        w_cmds = mixer.compute_motor_speed(u_virtual)

        # Clean rows are sampled from the pre-step state so features and the analytic
        # label refer to the exact same instant, just like dynamic_node calls the
        # predictor right before integrating.
        if clean and tick % log_stride == 0:
            feat = build_features(state, curr_wind, rotor_params)
            label = analytic_residual(state, ac_params, nominal_params, curr_wind, w_cmds)
            clean_rows.append([
                t_sim,
                float(state[9]),
                float(state[10]),
                float(-state[11]),
                float(state[6] * RAD_TO_DEG),
                float(state[7] * RAD_TO_DEG),
                float(state[8] * RAD_TO_DEG),
                *feat.tolist(),
                *label.tolist(),
            ])

        dx = six_dof_model(state, ac_params, curr_wind, w_cmds)
        state = forward_euler(state, dx, dt)

        # Normalize Euler angles via quaternion to match ROS 2 Odometry transport
        rot = Rotation.from_euler('xyz', state[6:9])
        roll, pitch, yaw = rot.as_euler('xyz')

        ctrl_state[0:6] = state[0:6]
        ctrl_state[6:9] = roll, pitch, yaw
        ctrl_state[9:12] = state[9:12]
        controller.update_state(ctrl_state)

        curr_position[0:3] = state[9:12]

        # Step trajectory FSM at 100 Hz
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
                pos_ref, vel_ref, yaw_ref, _ = path_manager.get_desired_state(
                    sample_t
                )
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
                error_pos = float(np.linalg.norm(curr_position - target_pos))
                if error_pos < wp_tolerance:
                    stabilize_time += dt_traj
                    if stabilize_time >= 2.0:
                        if segment_idx < (len(path_manager.segments) - 1):
                            segment_idx += 1
                            traj_time = path_manager.start_times[segment_idx]
                            mission_state = STATE_TRACKING
                        else:
                            if abs(curr_position[2]) < 0.25:
                                mission_state = STATE_IDLE
                            else:
                                last_wp = path_manager.waypoints[-1]
                                start_p = np.array([
                                    last_wp[0], last_wp[1], curr_position[2]
                                ])
                                ground_p = np.array([
                                    last_wp[0], last_wp[1], 0.0
                                ])
                                l_dist = float(np.linalg.norm(start_p - ground_p))
                                recovery_duration = max(4.0, l_dist / 1.5)
                                recovery_traj = PolynomialTrajectory(
                                    start_p, ground_p, recovery_duration
                                )
                                recovery_traj.generate_path()
                                recovery_time = 0.0
                                mission_state = STATE_LANDING
                else:
                    stabilize_time = 0.0

            elif mission_state == STATE_LANDING:
                recovery_time += dt_traj
                pos_ref = recovery_traj.sample_position(recovery_time)
                vel_ref = recovery_traj.sample_velocity(recovery_time)
                yaw_ref = 0.0
                if recovery_time >= recovery_duration:
                    mission_state = STATE_IDLE
            else:
                pos_ref = curr_position.copy()
                vel_ref = np.zeros(3)
                yaw_ref = 0.0

            waypoint_cmd[0:3] = pos_ref
            waypoint_cmd[3] = yaw_ref
            controller.desired_state(waypoint_cmd, vel_ref)

        # Multi-rate PX4 cascade control loop (50 Hz / 250 Hz / 1000 Hz)
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

        if not clean and tick % log_stride == 0:
            rows.append([
                t_sim,
                float(state[9]),
                float(state[10]),
                float(-state[11]),
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
            ])

        if mission_state == STATE_IDLE:
            break

    if clean:
        rows = clean_rows
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    data_matrix = np.asarray(rows, dtype=np.float64)
    # Residual labels can be ~1e-4, so the clean CSV keeps more significant digits.
    np.savetxt(
        output_csv_path,
        data_matrix,
        delimiter=',',
        header=','.join(CLEAN_CSV_HEADER if clean else CSV_HEADER),
        comments='',
        fmt='%.9g' if clean else '%.6f',
    )
    try:
        os.chmod(output_csv_path, 0o666)
    except OSError:
        pass

    final_time = rows[-1][0] if rows else 0.0
    return mission_path.stem, final_time, len(rows)


_Task = tuple[AircraftConfig, Path, Path, float, AircraftConfig | None]


def _worker_task(task: _Task) -> tuple[str, Path, float, int]:
    """Execute ``simulate_mission`` inside a worker process."""
    ac_cfg, mission_path, output_csv_path, log_hz, nominal_cfg = task
    stem, sim_t, n_rows = simulate_mission(
        ac_cfg, mission_path, output_csv_path, log_hz=log_hz, nominal_cfg=nominal_cfg
    )
    return stem, output_csv_path, sim_t, n_rows


def collect_mission_tasks(
    ac_cfg: AircraftConfig,
    target_sub: str,
    splits: list[str],
    log_hz: float,
    nominal_cfg: AircraftConfig | None = None,
) -> list[_Task]:
    """Build the list of mission simulation tasks across splits and wind conditions.

    Args:
        ac_cfg: Loaded aircraft configuration.
        target_sub: Output subfolder under ``data/flight_logs`` (e.g. ``'real'`` or ``'sim'``).
        splits: Dataset splits to run (subset of ``['training', 'validation', 'test']``).
        log_hz: Logging frequency [Hz].
        nominal_cfg: Nominal model for computing exact analytic residual labels.

    Returns:
        List of worker task tuples
        ``(ac_cfg, mission_path, output_csv_path, log_hz, nominal_cfg)``.
    """
    split_folder_map = {
        'training': 'train',
        'validation': 'val',
        'test': 'test',
    }
    aircraft_slug = ac_cfg.name.strip().replace(' ', '_').lower()
    tasks: list[_Task] = []

    for split_name in splits:
        split_dir = paths.MISSIONS_DIR / split_name
        if not split_dir.exists():
            continue

        out_split_name = split_folder_map.get(split_name, split_name)
        out_dir = paths.FLIGHT_LOGS_DIR / target_sub / out_split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(out_dir.parent, 0o777)
            os.chmod(out_dir, 0o777)
        except OSError:
            pass

        for cond in ('nowind', 'wind'):
            cond_dir = split_dir / cond
            if not cond_dir.exists():
                continue
            for mission_path in sorted(cond_dir.glob('*.json')):
                csv_name = f'{aircraft_slug}_{mission_path.stem}.csv'
                out_csv = out_dir / csv_name
                tasks.append((ac_cfg, mission_path, out_csv, log_hz, nominal_cfg))

    return tasks


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the headless batch flight simulation."""
    parser = argparse.ArgumentParser(
        description=(
            'Run offline batch simulations for all train/val/test missions '
            'and save CSV flight logs under data/flight_logs/<sim|real>/<train|val|test>.'
        )
    )
    parser.add_argument(
        '-a',
        '--aircraft',
        default='real',
        help=(
            "Aircraft model to simulate: 'real' (m350_model_real.json), "
            "'sim' (m350_model.json), 'both', or path/name of a JSON model "
            "(default: 'real')."
        ),
    )
    parser.add_argument(
        '-s',
        '--splits',
        nargs='+',
        default=['training', 'validation', 'test'],
        choices=['training', 'validation', 'test'],
        help='Mission splits to simulate (default: training validation test).',
    )
    parser.add_argument(
        '--log-hz',
        type=float,
        default=100.0,
        help='Telemetry sampling rate in Hz for the exported CSV logs (default: 100.0).',
    )
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=os.cpu_count() or 4,
        help='Number of parallel worker processes (default: all CPU cores).',
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        default=True,
        help=(
            'Log rotor-based features and exact analytic residual labels '
            '(aircraft minus m350_model.json) to data/flight_logs/<sub> (default: True).'
        ),
    )
    args = parser.parse_args(argv)

    nominal_cfg = load_aircraft_config('sim')[0] if args.clean else None
    aircraft_list = (
        ['sim', 'real'] if args.aircraft.lower() == 'both' else [args.aircraft]
    )

    for ac_arg in aircraft_list:
        ac_cfg, target_sub = load_aircraft_config(ac_arg)
        tasks = collect_mission_tasks(
            ac_cfg, target_sub, args.splits, args.log_hz, nominal_cfg
        )

        if not tasks:
            print(f'No missions found for splits: {args.splits}')
            return 1

        out_root = paths.FLIGHT_LOGS_DIR / target_sub
        print(
            f'=== Batch Simulation: {ac_cfg.name} ({len(tasks)} missions) ===\n'
            f'Output directory: {out_root}/{{train,val,test}}\n'
            f'Workers: {args.workers} | Log rate: {args.log_hz:.1f} Hz'
        )

        t0 = time.perf_counter()
        completed = 0
        total_sim_seconds = 0.0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker_task, t): t for t in tasks}
            for future in as_completed(futures):
                completed += 1
                stem, out_csv, sim_t, n_rows = future.result()
                total_sim_seconds += sim_t
                rel_out = out_csv.relative_to(paths.PROJECT_ROOT)
                print(
                    f'  [{completed:02d}/{len(tasks):02d}] {stem:<8} -> '
                    f'{rel_out} ({sim_t:.1f}s simulated, {n_rows} samples)'
                )

        elapsed = time.perf_counter() - t0
        speedup = total_sim_seconds / max(elapsed, 1e-6)
        print(
            f'Completed {len(tasks)} missions ({total_sim_seconds / 60.0:.1f} min '
            f'of flight) in {elapsed:.2f}s ({speedup:.0f}x real-time).\n'
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
