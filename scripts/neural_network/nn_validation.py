"""Validation, unseen test evaluation, and comparative plotting for 6-DoF MLPs.

Loads all candidate MLP architectures trained by ``nn_training.py``, evaluates
their generalization on the validation and unseen test splits (both calm air
and turbulent Dryden/sinusoidal wind flights), and generates comparative diagnostic
figures in ``data/nn_validation``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_m300')

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_CURRENT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR), str(_CURRENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_network.nn_training import (  # noqa: E402
    DroneDataset,
    DroneResidualMLP,
    TARGET_COLS,
    compute_metrics,
)
import paths  # noqa: E402

AXIS_LABELS = [
    r'$\Delta a_u$ [m/s²] (Longitudinal)',
    r'$\Delta a_v$ [m/s²] (Lateral)',
    r'$\Delta a_w$ [m/s²] (Vertical)',
    r'$\Delta \alpha_p$ [rad/s²] (Roll)',
    r'$\Delta \alpha_q$ [rad/s²] (Pitch)',
    r'$\Delta \alpha_r$ [rad/s²] (Yaw)',
]

SHORT_AXIS_NAMES = [
    r'$\Delta a_u$',
    r'$\Delta a_v$',
    r'$\Delta a_w$',
    r'$\Delta \alpha_p$',
    r'$\Delta \alpha_q$',
    r'$\Delta \alpha_r$',
]

PALETTE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
]


def plot_loss_curves(models: list[dict[str, Any]], out_dir: Path) -> Path:
    """Plot training and validation MSE convergence curves for all architectures."""
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(15, 6))

    for i, m in enumerate(models):
        color = PALETTE[i % len(PALETTE)]
        epochs = np.arange(1, len(m['train_losses']) + 1)
        label = f'{m["name"]} ({m["num_parameters"] // 1000}k params)'
        lw = 2.5 if i == 0 else 1.6
        ax_train.plot(
            epochs, m['train_losses'], label=label, color=color, linewidth=lw
        )
        ax_val.plot(
            epochs, m['val_losses'], label=label, color=color, linewidth=lw
        )
        best_ep = m['best_epoch']
        best_val = m['val_losses'][best_ep - 1]
        ax_val.scatter([best_ep], [best_val], color=color, s=45, zorder=5)

    ax_train.set_title('Training Loss Convergence (Normalized MSE)', fontsize=13)
    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Normalized MSE')
    ax_train.set_yscale('log')
    ax_train.grid(True, which='both', alpha=0.3)
    ax_train.legend(fontsize=9)

    ax_val.set_title('Validation Loss Convergence (Normalized MSE)', fontsize=13)
    ax_val.set_xlabel('Epoch')
    ax_val.set_ylabel('Normalized MSE')
    ax_val.set_yscale('log')
    ax_val.grid(True, which='both', alpha=0.3)
    ax_val.legend(fontsize=9)

    fig.suptitle(
        'MLP Architecture Search: Training & Validation Convergence',
        fontsize=15,
        fontweight='bold',
    )
    fig.tight_layout()
    out_path = out_dir / '01_training_val_loss_curves.png'
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_architecture_comparison(
    models: list[dict[str, Any]], out_dir: Path
) -> Path:
    """Generate a 4-panel bar chart comparing test accuracy and model complexity."""
    names = [m['name'].replace('MLP_', '') for m in models]
    x = np.arange(len(names))

    val_r2 = [m['val_metrics']['overall_r2'] for m in models]
    test_r2 = [m['test_metrics']['overall_r2'] for m in models]
    lin_rmse = [m['test_metrics']['linear_rmse_ms2'] for m in models]
    ang_rmse = [m['test_metrics']['angular_rmse_rads2'] for m in models]
    params_k = [m['num_parameters'] / 1000.0 for m in models]
    smooth_rms = [m['test_metrics']['smoothness_rms'] for m in models]

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    # Panel 1: R² Score (Val vs Test)
    w = 0.36
    axs[0, 0].bar(
        x - w / 2, val_r2, width=w, label='Validation R²', color='#2b7bba'
    )
    bars_r2 = axs[0, 0].bar(
        x + w / 2, test_r2, width=w, label='Unseen Test R²', color='#2ca02c'
    )
    axs[0, 0].set_title('Overall R² Coefficient of Determination (Higher is Better)')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(names, rotation=20, ha='right')
    axs[0, 0].set_ylabel('R² Score')
    min_r2 = max(0.0, min(min(val_r2), min(test_r2)) - 0.05)
    axs[0, 0].set_ylim(min_r2, 1.01)
    axs[0, 0].grid(True, axis='y', alpha=0.3)
    axs[0, 0].legend()
    for b in bars_r2:
        h_val = b.get_height()
        axs[0, 0].text(
            b.get_x() + b.get_width() / 2,
            h_val + 0.003,
            f'{h_val:.4f}',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

    # Panel 2: Linear Acceleration Test RMSE [m/s²]
    colors_lin = ['#2ca02c' if i == 0 else '#4a90d9' for i in range(len(models))]
    bars_lin = axs[0, 1].bar(x, lin_rmse, color=colors_lin, width=0.55)
    axs[0, 1].set_title('Unseen Test: Linear Residual RMSE [m/s²] (Lower is Better)')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(names, rotation=20, ha='right')
    axs[0, 1].set_ylabel('RMSE [m/s²]')
    axs[0, 1].grid(True, axis='y', alpha=0.3)
    for b in bars_lin:
        h_val = b.get_height()
        axs[0, 1].text(
            b.get_x() + b.get_width() / 2,
            h_val * 1.01,
            f'{h_val:.4f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    # Panel 3: Angular Acceleration Test RMSE [rad/s²]
    colors_ang = ['#2ca02c' if i == 0 else '#e67e22' for i in range(len(models))]
    bars_ang = axs[1, 0].bar(x, ang_rmse, color=colors_ang, width=0.55)
    axs[1, 0].set_title(
        'Unseen Test: Angular Residual RMSE [rad/s²] (Lower is Better)'
    )
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(names, rotation=20, ha='right')
    axs[1, 0].set_ylabel('RMSE [rad/s²]')
    axs[1, 0].grid(True, axis='y', alpha=0.3)
    for b in bars_ang:
        h_val = b.get_height()
        axs[1, 0].text(
            b.get_x() + b.get_width() / 2,
            h_val * 1.01,
            f'{h_val:.4f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    # Panel 4: Parameter Count vs CAPS Temporal Smoothness
    ax4 = axs[1, 1]
    ax4_twin = ax4.twinx()
    b1 = ax4.bar(
        x - w / 2,
        params_k,
        width=w,
        color='#7f8c8d',
        label='Parameters (×10³)',
    )
    b2 = ax4_twin.bar(
        x + w / 2,
        smooth_rms,
        width=w,
        color='#8e44ad',
        label='Step Variation RMS (CAPS)',
    )
    ax4.set_title('Model Capacity vs. Temporal Smoothness on Test Set')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=20, ha='right')
    ax4.set_ylabel('Trainable Parameters [×10³]', color='#555555')
    ax4_twin.set_ylabel('Step-to-Step Residual Δ RMS', color='#8e44ad')
    ax4.grid(True, axis='y', alpha=0.3)
    lines = [b1[0], b2[0]]
    labels = ['Parameters (×10³)', 'Step Variation RMS (CAPS)']
    ax4.legend(lines, labels, loc='upper left')

    fig.suptitle(
        f'MLP Architecture Benchmark on Unseen Test Flights '
        f'(Winner: {models[0]["name"]})',
        fontsize=15,
        fontweight='bold',
    )
    fig.tight_layout()
    out_path = out_dir / '02_architecture_comparison_metrics.png'
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_per_axis_comparison(
    models: list[dict[str, Any]], out_dir: Path
) -> Path:
    """Plot per-axis RMSE and R² across all 6 candidate MLP architectures."""
    fig, (ax_rmse, ax_r2) = plt.subplots(2, 1, figsize=(15, 10))
    n_models = len(models)
    x = np.arange(len(TARGET_COLS))
    width = 0.8 / max(1, n_models)

    for i, m in enumerate(models):
        offset = (i - (n_models - 1) / 2.0) * width
        rmse_vals = [
            m['test_metrics']['per_axis_rmse'][c] for c in TARGET_COLS
        ]
        r2_vals = [m['test_metrics']['per_axis_r2'][c] for c in TARGET_COLS]
        label = m['name'].replace('MLP_', '')
        color = PALETTE[i % len(PALETTE)]
        ax_rmse.bar(x + offset, rmse_vals, width=width, label=label, color=color)
        ax_r2.bar(x + offset, r2_vals, width=width, label=label, color=color)

    ax_rmse.set_title(
        'Per-Axis Residual RMSE on Unseen Test Set (Lower is Better)',
        fontsize=13,
    )
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(AXIS_LABELS)
    ax_rmse.set_ylabel('RMSE ([m/s²] or [rad/s²])')
    ax_rmse.grid(True, axis='y', alpha=0.3)
    ax_rmse.legend(ncol=3, fontsize=9)

    ax_r2.set_title(
        'Per-Axis R² Score on Unseen Test Set (Higher is Better)',
        fontsize=13,
    )
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(AXIS_LABELS)
    ax_r2.set_ylabel('R² Score')
    ax_r2.set_ylim(0.0, 1.05)
    ax_r2.grid(True, axis='y', alpha=0.3)
    ax_r2.legend(ncol=3, fontsize=9)

    fig.suptitle(
        '6-DoF Axis-by-Axis Generalization Across MLP Architectures',
        fontsize=15,
        fontweight='bold',
    )
    fig.tight_layout()
    out_path = out_dir / '03_per_axis_r2_rmse_comparison.png'
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_6dof_flight_tracking(
    flight_title: str,
    y_true: np.ndarray,
    y_best: np.ndarray,
    y_baseline: np.ndarray,
    best_name: str,
    baseline_name: str,
    out_path: Path,
) -> Path:
    """Plot a 3x2 6-DoF time-series comparison for a single unseen test flight."""
    t_sec = np.arange(len(y_true)) * 0.01
    fig, axs = plt.subplots(3, 2, figsize=(16, 11), sharex=True)

    for i in range(3):
        # Left column: linear accelerations (0, 1, 2)
        ax_l = axs[i, 0]
        ax_l.plot(
            t_sec,
            y_true[:, i],
            color='#1f77b4',
            alpha=0.65,
            linewidth=1.3,
            label='Ground-Truth Residual (M350 Real)',
        )
        ax_l.plot(
            t_sec,
            y_baseline[:, i],
            color='#ff7f0e',
            linestyle=':',
            alpha=0.8,
            linewidth=1.2,
            label=f'{baseline_name}',
        )
        ax_l.plot(
            t_sec,
            y_best[:, i],
            color='#d62728',
            linestyle='--',
            alpha=0.95,
            linewidth=1.5,
            label=f'{best_name} (Best)',
        )
        ax_l.set_ylabel(AXIS_LABELS[i])
        ax_l.grid(True, alpha=0.3)
        if i == 0:
            ax_l.set_title('Translational Acceleration Residuals (Body Frame)')
            ax_l.legend(loc='upper right', fontsize=8)
        if i == 2:
            ax_l.set_xlabel('Flight Time [s]')

        # Right column: angular accelerations (3, 4, 5)
        idx = i + 3
        ax_r = axs[i, 1]
        ax_r.plot(
            t_sec,
            y_true[:, idx],
            color='#1f77b4',
            alpha=0.65,
            linewidth=1.3,
            label='Ground-Truth Residual (M350 Real)',
        )
        ax_r.plot(
            t_sec,
            y_baseline[:, idx],
            color='#ff7f0e',
            linestyle=':',
            alpha=0.8,
            linewidth=1.2,
            label=f'{baseline_name}',
        )
        ax_r.plot(
            t_sec,
            y_best[:, idx],
            color='#d62728',
            linestyle='--',
            alpha=0.95,
            linewidth=1.5,
            label=f'{best_name} (Best)',
        )
        ax_r.set_ylabel(AXIS_LABELS[idx])
        ax_r.grid(True, alpha=0.3)
        if i == 0:
            ax_r.set_title('Rotational Acceleration Residuals (Body Frame)')
            ax_r.legend(loc='upper right', fontsize=8)
        if i == 2:
            ax_r.set_xlabel('Flight Time [s]')

    fig.suptitle(flight_title, fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_zoom_and_parity(
    y_true: np.ndarray,
    y_best: np.ndarray,
    best_name: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Generate zoomed maneuver tracking and 6-DoF parity scatter plots."""
    # 1. Zoomed window (20s to 55s of unseen wind test flight)
    n_total = len(y_true)
    i0 = min(2000, max(0, n_total // 5))
    i1 = min(n_total, i0 + 3000)
    t_zoom = np.arange(i0, i1) * 0.01

    fig_z, axs_z = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    for i in range(3):
        axs_z[i, 0].plot(
            t_zoom,
            y_true[i0:i1, i],
            color='#1f77b4',
            linewidth=1.6,
            label='Ground-Truth Residual',
        )
        axs_z[i, 0].plot(
            t_zoom,
            y_best[i0:i1, i],
            color='#d62728',
            linestyle='--',
            linewidth=1.6,
            label=f'{best_name} Prediction',
        )
        axs_z[i, 0].set_ylabel(AXIS_LABELS[i])
        axs_z[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axs_z[i, 0].set_title('Zoomed Linear Residual Tracking (30 s Window)')
            axs_z[i, 0].legend(loc='upper right', fontsize=9)
        if i == 2:
            axs_z[i, 0].set_xlabel('Flight Time [s]')

        idx = i + 3
        axs_z[i, 1].plot(
            t_zoom,
            y_true[i0:i1, idx],
            color='#1f77b4',
            linewidth=1.6,
            label='Ground-Truth Residual',
        )
        axs_z[i, 1].plot(
            t_zoom,
            y_best[i0:i1, idx],
            color='#d62728',
            linestyle='--',
            linewidth=1.6,
            label=f'{best_name} Prediction',
        )
        axs_z[i, 1].set_ylabel(AXIS_LABELS[idx])
        axs_z[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axs_z[i, 1].set_title('Zoomed Angular Residual Tracking (30 s Window)')
            axs_z[i, 1].legend(loc='upper right', fontsize=9)
        if i == 2:
            axs_z[i, 1].set_xlabel('Flight Time [s]')

    fig_z.suptitle(
        f'High-Resolution Maneuver Tracking ({best_name})',
        fontsize=15,
        fontweight='bold',
    )
    fig_z.tight_layout()
    zoom_path = out_dir / '06_best_model_maneuver_zoom.png'
    fig_z.savefig(zoom_path, dpi=160)
    plt.close(fig_z)

    # 2. Parity scatter plots across all unseen test samples
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        len(y_true), size=min(6000, len(y_true)), replace=False
    )

    fig_p, axs_p = plt.subplots(2, 3, figsize=(16, 10))
    for idx in range(6):
        r, c = divmod(idx, 3)
        ax = axs_p[r, c]
        yt = y_true[sample_idx, idx]
        yp = y_best[sample_idx, idx]
        ax.scatter(yt, yp, s=6, alpha=0.25, color='#2b7bba', edgecolors='none')
        low = float(min(np.min(yt), np.min(yp)))
        high = float(max(np.max(yt), np.max(yp)))
        ax.plot([low, high], [low, high], 'r--', linewidth=1.5, label='Ideal y = x')
        ax.set_title(AXIS_LABELS[idx], fontsize=11)
        ax.set_xlabel('True Residual')
        ax.set_ylabel('Predicted Residual')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    fig_p.suptitle(
        f'Unseen Test Set Parity Plots: True vs. Predicted Residuals ({best_name})',
        fontsize=15,
        fontweight='bold',
    )
    fig_p.tight_layout()
    parity_path = out_dir / '07_parity_and_error_distribution.png'
    fig_p.savefig(parity_path, dpi=160)
    plt.close(fig_p)

    return zoom_path, parity_path


def main(argv: list[str] | None = None) -> int:
    """Evaluate all trained MLP architectures and generate comparative plots."""
    parser = argparse.ArgumentParser(
        description='Validate and compare trained 6-DoF residual MLP models.'
    )
    parser.add_argument(
        '--domain',
        default='real',
        choices=['real', 'sim'],
        help="Flight log domain under data/flight_logs to evaluate (default: 'real').",
    )
    args = parser.parse_args(argv)

    summary_path = paths.NN_MODELS_DIR / 'models_summary.json'
    if not summary_path.exists():
        print(
            f'Summary file not found: {summary_path}. '
            'Run scripts/neural_network/nn_training.py first.'
        )
        return 1

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    window_h = int(summary.get('window_h', 1))
    feature_cols: list[str] | None = summary.get('feature_columns')
    models: list[dict[str, Any]] = summary['models']

    input_mean = np.load(paths.NN_MODELS_DIR / 'input_mean.npy')
    input_std = np.load(paths.NN_MODELS_DIR / 'input_std.npy')
    target_mean = np.load(paths.NN_MODELS_DIR / 'target_mean.npy')
    target_std = np.load(paths.NN_MODELS_DIR / 'target_std.npy')

    log_base = paths.FLIGHT_LOGS_DIR / args.domain
    test_files = sorted((log_base / 'test').glob('*.csv'))
    test_ds = DroneDataset(
        test_files,
        feature_cols=feature_cols,
        h=window_h,
        stats=(input_mean, input_std, target_mean, target_std),
    )

    best_info = models[0]
    baseline_info = models[-1]

    best_model = DroneResidualMLP.load(
        paths.PROJECT_ROOT / best_info['checkpoint_path']
    )
    baseline_model = DroneResidualMLP.load(
        paths.PROJECT_ROOT / baseline_info['checkpoint_path']
    )

    y_test_true = test_ds.Y * target_std + target_mean
    y_test_best = best_model.forward(test_ds.X) * target_std + target_mean
    y_test_base = baseline_model.forward(test_ds.X) * target_std + target_mean

    out_dir = paths.NN_VALIDATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.append(plot_loss_curves(models, out_dir))
    generated.append(plot_architecture_comparison(models, out_dir))
    generated.append(plot_per_axis_comparison(models, out_dir))

    # Select one wind test flight and one calm-air test flight for time-series plots
    wind_slice = next(
        (s for s in test_ds.file_slices if s[0].endswith('_w')),
        test_ds.file_slices[-1],
    )
    nowind_slice = next(
        (s for s in test_ds.file_slices if not s[0].endswith('_w')),
        test_ds.file_slices[0],
    )

    w_name, w_start, w_end = wind_slice
    generated.append(
        plot_6dof_flight_tracking(
            flight_title=(
                f'Unseen Wind Test Flight ({w_name}): '
                f'{best_info["name"]} vs. {baseline_info["name"]}'
            ),
            y_true=y_test_true[w_start:w_end],
            y_best=y_test_best[w_start:w_end],
            y_baseline=y_test_base[w_start:w_end],
            best_name=best_info['name'],
            baseline_name=baseline_info['name'],
            out_path=out_dir / '04_best_model_6dof_test_wind.png',
        )
    )

    nw_name, nw_start, nw_end = nowind_slice
    generated.append(
        plot_6dof_flight_tracking(
            flight_title=(
                f'Unseen Calm-Air Test Flight ({nw_name}): '
                f'{best_info["name"]} vs. {baseline_info["name"]}'
            ),
            y_true=y_test_true[nw_start:nw_end],
            y_best=y_test_best[nw_start:nw_end],
            y_baseline=y_test_base[nw_start:nw_end],
            best_name=best_info['name'],
            baseline_name=baseline_info['name'],
            out_path=out_dir / '05_best_model_6dof_test_nowind.png',
        )
    )

    z_path, p_path = plot_zoom_and_parity(
        y_true=y_test_true[w_start:w_end],
        y_best=y_test_best[w_start:w_end],
        best_name=best_info['name'],
        out_dir=out_dir,
    )
    generated.extend([z_path, p_path])

    for fig_file in generated:
        try:
            os.chmod(fig_file, 0o666)
        except OSError:
            pass

    best_test_metrics = compute_metrics(y_test_true, y_test_best)
    print(f'=== Validation & Test Report ({args.domain.upper()}) ===')
    print(f'Best Architecture : {best_info["name"]} ({best_info["full_dims"]})')
    print(f'Overall Test R²   : {best_test_metrics["overall_r2"]:.4f}')
    print(f'Linear Test RMSE  : {best_test_metrics["linear_rmse_ms2"]:.4f} m/s²')
    print(f'Angular Test RMSE : {best_test_metrics["angular_rmse_rads2"]:.4f} rad/s²')
    print('Generated Figures :')
    for p in generated:
        print(f'  - {p.relative_to(paths.PROJECT_ROOT)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
