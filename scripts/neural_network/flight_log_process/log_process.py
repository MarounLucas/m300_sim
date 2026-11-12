"""Flight log filtering and 6-DoF structural residual acceleration processor.

Reads raw telemetry CSV files from ``data/flight_logs/<aircraft>/<split>``,
trims idle ground segments, applies zero-phase Butterworth low-pass filtering
to eliminate multi-rate control aliasing, computes the nominal 6-DoF
accelerations predicted by ``m350_model.json`` (including nominal aerodynamic
drag and rotor-matched actuation), extracts the structural 6-DoF model-plant
residuals and trigonometric attitude encodings, and writes the filtered
datasets to ``data/flight_logs/<aircraft>/ftr/<split>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

_CURRENT_DIR = Path(__file__).resolve().parent
_NN_DIR = _CURRENT_DIR.parent
_SCRIPTS_DIR = _NN_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths  # noqa: E402

RESIDUAL_COLS = [
    'Delta_a_u',
    'Delta_a_v',
    'Delta_a_w',
    'Delta_alpha_p',
    'Delta_alpha_q',
    'Delta_alpha_r',
]

ANGLE_COLS = ['Roll_deg', 'Pitch_deg', 'Yaw_deg']


def load_aircraft_dict(model_path: Path) -> dict[str, float]:
    """Load mass, inertia, drag, and propulsion parameters from an aircraft JSON file.

    Args:
        model_path: Path to the aircraft JSON configuration file.

    Returns:
        Dictionary with ``mass`` [kg], ``jx``, ``jy``, ``jz``, ``jxz`` [kg·m²],
        ``cd`` [N/(m/s)], ``k_t0`` [N/(rad/s)²], ``k_q0`` [N·m/(rad/s)²], and ``g`` [m/s²].
    """
    if model_path.exists():
        with open(model_path, 'r', encoding='utf-8') as f:
            raw: dict[str, Any] = json.load(f)
        return {
            'mass': float(raw['mass']),
            'jx': float(raw['jx']),
            'jy': float(raw['jy']),
            'jz': float(raw['jz']),
            'jxz': float(raw.get('jxz', 0.0)),
            'cd': float(raw.get('cd', 0.25)),
            'k_t0': float(raw.get('k_t0', 0.0002433)),
            'k_q0': float(raw.get('k_q0', 6.92e-06)),
            'g': 9.81,
        }
    return {
        'mass': 6.47,
        'jx': 0.214,
        'jy': 0.2,
        'jz': 0.389,
        'jxz': 0.0,
        'cd': 0.25,
        'k_t0': 0.0002433,
        'k_q0': 6.92e-06,
        'g': 9.81,
    }


def load_nominal_params(model_path: Path | None = None) -> dict[str, float]:
    """Load nominal aircraft parameters from ``m350_model.json``."""
    target_path = model_path or (paths.AIRCRAFT_MODELS_DIR / 'm350_model.json')
    return load_aircraft_dict(target_path)


def process_single_log(
    input_filepath: Path,
    output_filepath: Path,
    nominal_params: dict[str, float],
    plant_params: dict[str, float] | None = None,
) -> tuple[bool, int]:
    """Filter a single flight log and compute 6-DoF structural model-plant residuals.

    Applies a 4th-order zero-phase Butterworth low-pass filter (cutoff 6 Hz at
    100 Hz sampling) to remove 250 Hz / 1000 Hz multi-rate control aliasing from
    the logged wrenches and extracts the structural acceleration residuals
    compatible with ``six_dof_model``.

    Args:
        input_filepath: Path to the raw flight log CSV.
        output_filepath: Path where the filtered CSV with residual columns is saved.
        nominal_params: Nominal aircraft parameters dictionary (``m350_model.json``).
        plant_params: Optional plant parameters dictionary (``m350_model_real.json``).

    Returns:
        Tuple ``(success, num_filtered_rows)``.
    """
    with open(input_filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False, 0
        raw_rows = [[float(v) for v in row] for row in reader if row]

    if not raw_rows:
        return False, 0

    data = np.asarray(raw_rows, dtype=np.float64)
    col_idx = {name: i for i, name in enumerate(header)}

    pos_alt = data[:, col_idx['Pos_Alt_m']]
    u_all = data[:, col_idx['Vel_u_ms']]
    v_all = data[:, col_idx['Vel_v_ms']]
    w_all = data[:, col_idx['Vel_w_ms']]

    v_linear = np.sqrt(u_all**2 + v_all**2 + w_all**2)
    mask_active = (pos_alt > 0.05) & (v_linear > 0.05)

    if not np.any(mask_active):
        print(
            f'  -> WARNING: No dynamic flight detected in {input_filepath.name}. Skipping.'
        )
        return False, 0

    active_indices = np.flatnonzero(mask_active)
    t_start_idx = int(active_indices[0])
    t_end_idx = int(active_indices[-1])
    trimmed = data[t_start_idx : t_end_idx + 1].copy()

    if len(trimmed) < 32:
        return False, 0

    # Zero-phase 4th-order Butterworth low-pass filter (6 Hz cutoff at 100 Hz sampling)
    # removes the 50 Hz aliasing sawtooth between the 250 Hz attitude loop and 100 Hz log.
    b_lp, a_lp = butter(4, 6.0 / 50.0, btype='low')

    mass_s = nominal_params['mass']
    jx_s = nominal_params['jx']
    jy_s = nominal_params['jy']
    jz_s = nominal_params['jz']
    jxz_s = nominal_params['jxz']
    cd_s = nominal_params['cd']
    g = nominal_params.get('g', 9.81)

    p_dict = plant_params or load_aircraft_dict(
        paths.AIRCRAFT_MODELS_DIR / 'm350_model_real.json'
    )
    mass_r = p_dict['mass']
    jx_r = p_dict['jx']
    jy_r = p_dict['jy']
    jz_r = p_dict['jz']
    jxz_r = p_dict['jxz']
    cd_r = p_dict['cd']

    phi = np.radians(trimmed[:, col_idx['Roll_deg']])
    theta = np.radians(trimmed[:, col_idx['Pitch_deg']])

    p = trimmed[:, col_idx['Rate_p_rads']]
    q = trimmed[:, col_idx['Rate_q_rads']]
    r = trimmed[:, col_idx['Rate_r_rads']]

    u = trimmed[:, col_idx['Vel_u_ms']]
    v = trimmed[:, col_idx['Vel_v_ms']]
    w = trimmed[:, col_idx['Vel_w_ms']]

    # Filter the logged control wrench to match physical rotor bandwidth
    fz_f = filtfilt(b_lp, a_lp, trimmed[:, col_idx['Thrust_Fz_N']])
    tx_f = filtfilt(b_lp, a_lp, trimmed[:, col_idx['Torque_Tx_Nm']])
    ty_f = filtfilt(b_lp, a_lp, trimmed[:, col_idx['Torque_Ty_Nm']])
    tz_f = filtfilt(b_lp, a_lp, trimmed[:, col_idx['Torque_Tz_Nm']])

    trimmed[:, col_idx['Thrust_Fz_N']] = fz_f
    trimmed[:, col_idx['Torque_Tx_Nm']] = tx_f
    trimmed[:, col_idx['Torque_Ty_Nm']] = ty_f
    trimmed[:, col_idx['Torque_Tz_Nm']] = tz_f

    ax_meas = trimmed[:, col_idx['Accel_x_ms2']]
    ay_meas = trimmed[:, col_idx['Accel_y_ms2']]
    az_meas = trimmed[:, col_idx['Accel_z_ms2']]
    ap_meas = trimmed[:, col_idx['Alpha_p_rads2']]
    aq_meas = trimmed[:, col_idx['Alpha_q_rads2']]
    ar_meas = trimmed[:, col_idx['Alpha_r_rads2']]

    # Recover instantaneous rotor torques from plant Newton-Euler equations so that
    # nominal vs. plant comparison is evaluated at identical rotor speeds without
    # sub-Nyquist 3 ms motor-step dipole spikes.
    tau_x_rot = jx_r * ap_meas - jxz_r * ar_meas - (jy_r - jz_r) * q * r - jxz_r * p * q
    tau_y_rot = jy_r * aq_meas - (jz_r - jx_r) * r * p + jxz_r * (p**2 - r**2)
    tau_z_rot_r = (
        -jxz_r * ap_meas + jz_r * ar_meas - (jx_r - jy_r) * p * q + jxz_r * q * r
    )
    tau_z_rot_nom = tau_z_rot_r

    den_s = (jx_s * jz_s) - (jxz_s**2)
    d_p_nom = (1.0 / den_s) * (
        (jz_s * (jy_s - jz_s) - jxz_s**2) * q * r
        + jxz_s * (jx_s - jy_s + jz_s) * p * q
        + jz_s * tau_x_rot
        + jxz_s * tau_z_rot_nom
    )
    d_q_nom = (1.0 / jy_s) * (
        ((jz_s - jx_s) * r * p) - (jxz_s * (p**2 - r**2)) + tau_y_rot
    )
    d_r_nom = (1.0 / den_s) * (
        (jx_s * (jx_s - jy_s) + jxz_s**2) * p * q
        + jxz_s * (jy_s - jx_s - jz_s) * q * r
        + jxz_s * tau_x_rot
        + jx_s * tau_z_rot_nom
    )

    # Structural translational residuals:
    # For nowind logs, Accel_x/y/z minus nominal six_dof_model (with nominal drag)
    # equals the exact structural drag + mass residual. For windy (_w) logs, WindManager
    # already injects wind into six_dof_model online, so we isolate the structural
    # aircraft difference free of unobservable stochastic Dryden gusts.
    is_wind_log = input_filepath.stem.endswith('_w')
    if not is_wind_log:
        gb_x = -g * np.sin(theta)
        gb_y = g * np.cos(theta) * np.sin(phi)
        gb_z = g * np.cos(theta) * np.cos(phi)
        d_u_nom = -(cd_s / mass_s) * u + gb_x - (q * w) + (v * r)
        d_v_nom = -(cd_s / mass_s) * v + gb_y + (p * w) - (u * r)
        # Reconstruct exact rotor thrust from plant vertical equation to avoid step spikes
        fz_rot = mass_r * (az_meas - gb_z + (p * v) - (u * q)) + cd_r * w
        d_w_nom = (fz_rot / mass_s) - (cd_s / mass_s) * w + gb_z - (p * v) + (u * q)
        raw_du = ax_meas - d_u_nom
        raw_dv = ay_meas - d_v_nom
        raw_dw = az_meas - d_w_nom
    else:
        raw_du = -((cd_r / mass_r) - (cd_s / mass_s)) * u
        raw_dv = -((cd_r / mass_r) - (cd_s / mass_s)) * v
        fz_rot = -fz_f
        raw_dw = (
            (fz_rot / mass_r) - (cd_r / mass_r) * w
            - ((fz_rot / mass_s) - (cd_s / mass_s) * w)
        )

    delta_a_u = filtfilt(b_lp, a_lp, raw_du)
    delta_a_v = filtfilt(b_lp, a_lp, raw_dv)
    delta_a_w = filtfilt(b_lp, a_lp, raw_dw)
    delta_alpha_p = filtfilt(b_lp, a_lp, ap_meas - d_p_nom)
    delta_alpha_q = filtfilt(b_lp, a_lp, aq_meas - d_q_nom)
    delta_alpha_r = filtfilt(b_lp, a_lp, ar_meas - d_r_nom)

    residuals = [
        delta_a_u,
        delta_a_v,
        delta_a_w,
        delta_alpha_p,
        delta_alpha_q,
        delta_alpha_r,
    ]

    trig_cols: list[np.ndarray] = []
    trig_names: list[str] = []
    for angle_name in ANGLE_COLS:
        angle_rad = np.radians(trimmed[:, col_idx[angle_name]])
        trig_cols.append(np.sin(angle_rad))
        trig_cols.append(np.cos(angle_rad))
        trig_names.append(f'sin_{angle_name}')
        trig_names.append(f'cos_{angle_name}')

    zscore_cols: list[np.ndarray] = []
    zscore_names: list[str] = []
    for col_name, res_arr in zip(RESIDUAL_COLS, residuals):
        std_val = float(np.std(res_arr, ddof=1)) if len(res_arr) > 1 else 0.0
        if std_val < 1e-12:
            std_val = 1e-6
        mean_val = float(np.mean(res_arr))
        zscore_cols.append((res_arr - mean_val) / std_val)
        zscore_names.append(f'{col_name}_zscore')

    extra_matrix = np.column_stack(residuals + trig_cols + zscore_cols)
    full_matrix = np.hstack([trimmed, extra_matrix])
    out_header = header + RESIDUAL_COLS + trig_names + zscore_names

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_filepath,
        full_matrix,
        delimiter=',',
        header=','.join(out_header),
        comments='',
        fmt='%.6f',
    )
    try:
        os.chmod(output_filepath.parent, 0o777)
        os.chmod(output_filepath, 0o666)
    except OSError:
        pass

    return True, len(full_matrix)


def process_split_directory(
    input_dir: Path,
    output_dir: Path,
    prefix: str,
    nominal_params: dict[str, float],
    plant_params: dict[str, float],
) -> tuple[int, int]:
    """Process all raw CSV flight logs inside a single split folder.

    Args:
        input_dir: Directory containing raw CSV logs (e.g. ``real/train``).
        output_dir: Destination directory for filtered logs (e.g. ``real/ftr/train``).
        prefix: Filename prefix added to processed files (e.g. ``'train'``, ``'val'``).
        nominal_params: Nominal aircraft parameters dictionary.
        plant_params: Plant aircraft parameters dictionary.

    Returns:
        Tuple ``(num_processed_files, total_filtered_rows)``.
    """
    if not input_dir.is_dir():
        print(f'Directory not found: {input_dir}')
        return 0, 0

    csv_files = sorted(
        f for f in input_dir.glob('*.csv') if not f.name.startswith('ftr_')
    )
    if not csv_files:
        print(f'No raw CSV files found in: {input_dir}')
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob('*.csv'):
        try:
            stale.unlink()
        except OSError:
            pass

    success_count = 0
    total_rows = 0
    for in_csv in csv_files:
        out_csv = output_dir / f'{prefix}_{in_csv.name}'
        ok, n_rows = process_single_log(
            in_csv, out_csv, nominal_params, plant_params
        )
        if ok:
            success_count += 1
            total_rows += n_rows

    print(
        f'  [{prefix.upper():<5}] {success_count}/{len(csv_files)} logs processed '
        f'-> {output_dir.relative_to(paths.PROJECT_ROOT)} ({total_rows:,} samples)'
    )
    return success_count, total_rows


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and filter train/val/test flight logs."""
    parser = argparse.ArgumentParser(
        description=(
            'Filter raw flight logs and compute 6-DoF nominal vs. plant '
            'structural residual accelerations across train, val, and test splits.'
        )
    )
    parser.add_argument(
        '-a',
        '--aircraft',
        default='real',
        choices=['real', 'sim', 'both'],
        help="Dataset domain under data/flight_logs to process (default: 'real').",
    )
    parser.add_argument(
        '-s',
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='Splits to process (default: train val test).',
    )
    args = parser.parse_args(argv)

    nominal_params = load_nominal_params()
    domains = ['real', 'sim'] if args.aircraft == 'both' else [args.aircraft]

    for domain in domains:
        plant_file = (
            'm350_model_real.json' if domain == 'real' else 'm350_model.json'
        )
        plant_params = load_aircraft_dict(paths.AIRCRAFT_MODELS_DIR / plant_file)
        base_dir = paths.FLIGHT_LOGS_DIR / domain
        print(
            f'=== Processing Flight Logs ({domain.upper()}) ===\n'
            f'Nominal reference: mass={nominal_params["mass"]} kg, '
            f'J=({nominal_params["jx"]}, {nominal_params["jy"]}, {nominal_params["jz"]}), '
            f'Cd={nominal_params["cd"]}'
        )
        total_files = 0
        total_samples = 0
        for split in args.splits:
            in_dir = base_dir / split
            out_dir = base_dir / 'ftr' / split
            n_files, n_rows = process_split_directory(
                in_dir, out_dir, split, nominal_params, plant_params
            )
            total_files += n_files
            total_samples += n_rows
        print(
            f'Completed {domain.upper()}: {total_files} files, '
            f'{total_samples:,} filtered samples.\n'
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
