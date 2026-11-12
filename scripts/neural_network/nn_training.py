"""Multi-architecture MLP training for 6-DoF quadcopter acceleration residuals.

Reads the CSV flight logs produced by ``batch_sim.py`` in
``data/flight_logs/<domain>/{train,val,test}`` with exact analytic
plant-minus-nominal residual labels, fits a physics-informed linear least-squares
baseline, trains multiple Multi-Layer Perceptron (MLP) candidate architectures
using a native AVX2/OpenMP engine, and saves checkpoints and summary metrics to
``data/nn_models``.

Typical usage::

    python3 scripts/neural_network/flight_log_process/batch_sim.py -a real
    python3 scripts/neural_network/nn_training.py --features base
    python3 scripts/neural_network/nn_training.py --features wind --activate
    python3 scripts/neural_network/nn_training.py --features base --input-noise 1.0
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_CURRENT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR), str(_CURRENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_network.flight_log_process.batch_sim import load_aircraft_config  # noqa: E402
from neural_network.nn_features import (  # noqa: E402
    FEATURE_SETS,
    LABEL_COLS,
    rotor_params_from_dynamics,
)
import paths  # noqa: E402

TARGET_COLS: list[str] = LABEL_COLS

# The analytic residual is a smooth, low-order function of the state (mass, inertia,
# drag and yaw-torque coefficient mismatches), so compact networks generalize best.
CLEAN_ARCHITECTURES: dict[str, list[int]] = {
    'MLP_2L_32_32': [32, 32],
    'MLP_2L_64_32': [64, 32],
    'MLP_2L_64_64': [64, 64],
    'MLP_3L_64_64_32': [64, 64, 32],
}

# Nominal 1-sigma sensor noise per feature, in physical units, scaled by --input-noise.
# Rotor wrench values assume ~1 % per-rotor thrust/torque error from ESC rpm telemetry
# (~0.16 N per rotor -> ~0.05 N·m on Tx/Ty through the arm lengths).
SENSOR_NOISE_STD: dict[str, float] = {
    'Vel_u_ms': 0.05,
    'Vel_v_ms': 0.05,
    'Vel_w_ms': 0.05,
    'Rate_p_rads': 0.01,
    'Rate_q_rads': 0.01,
    'Rate_r_rads': 0.01,
    'Rotor_Fz_N': 0.3,
    'Rotor_Tx_Nm': 0.05,
    'Rotor_Ty_Nm': 0.05,
    'Rotor_Tz_Nm': 0.01,
    'Wind_u_ms': 0.5,
    'Wind_v_ms': 0.5,
    'Wind_w_ms': 0.5,
}

_C_ENGINE_SOURCE = r"""
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static void forward_batch(
    int B,
    int n_layers,
    const int* dims,
    const float* X,
    float** W,
    float** b,
    float** A
) {
    const float* prev_A = X;
    for (int l = 0; l < n_layers; ++l) {
        int in_d = dims[l];
        int out_d = dims[l + 1];
        const float* W_l = W[l];
        const float* b_l = b[l];
        float* A_l = A[l];
        int is_last = (l == n_layers - 1);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < B; ++i) {
            const float* x_row = prev_A + i * in_d;
            float* a_row = A_l + i * out_d;
            for (int j = 0; j < out_d; ++j) {
                a_row[j] = b_l[j];
            }
            for (int k = 0; k < in_d; ++k) {
                float x_ik = x_row[k];
                const float* w_row = W_l + k * out_d;
                #pragma omp simd
                for (int j = 0; j < out_d; ++j) {
                    a_row[j] += x_ik * w_row[j];
                }
            }
            if (!is_last) {
                for (int j = 0; j < out_d; ++j) {
                    a_row[j] = tanhf(a_row[j]);
                }
            }
        }
        prev_A = A_l;
    }
}

float eval_dataset_mse(
    int N,
    int batch_size,
    int n_layers,
    const int* dims,
    const float* X,
    const float* Y,
    float** W,
    float** b,
    float* Y_pred_out
) {
    int in_dim = dims[0];
    int out_dim = dims[n_layers];
    float** A = (float**)malloc(n_layers * sizeof(float*));
    for (int l = 0; l < n_layers; ++l) {
        A[l] = (float*)malloc(batch_size * dims[l + 1] * sizeof(float));
    }

    double total_se = 0.0;
    for (int start = 0; start < N; start += batch_size) {
        int B = (start + batch_size <= N) ? batch_size : (N - start);
        const float* X_b = X + (size_t)start * in_dim;
        const float* Y_b = Y + (size_t)start * out_dim;
        forward_batch(B, n_layers, dims, X_b, W, b, A);

        const float* pred = A[n_layers - 1];
        if (Y_pred_out != NULL) {
            memcpy(
                Y_pred_out + (size_t)start * out_dim,
                pred,
                (size_t)B * out_dim * sizeof(float)
            );
        }
        double batch_se = 0.0;
        int total_elems = B * out_dim;
        #pragma omp parallel for reduction(+:batch_se) schedule(static)
        for (int idx = 0; idx < total_elems; ++idx) {
            double diff = (double)pred[idx] - (double)Y_b[idx];
            batch_se += diff * diff;
        }
        total_se += batch_se;
    }

    for (int l = 0; l < n_layers; ++l) {
        free(A[l]);
    }
    free(A);
    return (float)(total_se / ((double)N * out_dim));
}

float train_epoch_caps(
    int N,
    int batch_size,
    int n_layers,
    const int* dims,
    const float* X,
    const float* Y,
    float** W,
    float** b,
    float** mW,
    float** vW,
    float** mb,
    float** vb,
    int* step_counter,
    float lr,
    float weight_decay,
    float lambda_ts
) {
    int in_dim = dims[0];
    int out_dim = dims[n_layers];

    float** A = (float**)malloc(n_layers * sizeof(float*));
    float** dZ = (float**)malloc(n_layers * sizeof(float*));
    float** dW = (float**)malloc(n_layers * sizeof(float*));
    float** db = (float**)malloc(n_layers * sizeof(float*));

    for (int l = 0; l < n_layers; ++l) {
        int d_in = dims[l];
        int d_out = dims[l + 1];
        A[l] = (float*)malloc(batch_size * d_out * sizeof(float));
        dZ[l] = (float*)malloc(batch_size * d_out * sizeof(float));
        dW[l] = (float*)malloc(d_in * d_out * sizeof(float));
        db[l] = (float*)malloc(d_out * sizeof(float));
    }

    double epoch_mse_sum = 0.0;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;

    for (int start = 0; start < N; start += batch_size) {
        int B = (start + batch_size <= N) ? batch_size : (N - start);
        const float* X_b = X + (size_t)start * in_dim;
        const float* Y_b = Y + (size_t)start * out_dim;

        forward_batch(B, n_layers, dims, X_b, W, b, A);

        const float* pred = A[n_layers - 1];
        float* dz_last = dZ[n_layers - 1];
        float inv_elems = 2.0f / ((float)B * (float)out_dim);

        double batch_se = 0.0;
        int total_elems = B * out_dim;
        #pragma omp parallel for reduction(+:batch_se) schedule(static)
        for (int idx = 0; idx < total_elems; ++idx) {
            float diff = pred[idx] - Y_b[idx];
            batch_se += (double)diff * (double)diff;
            dz_last[idx] = diff * inv_elems;
        }
        epoch_mse_sum += batch_se / (double)out_dim;

        if (B > 1 && lambda_ts > 0.0f) {
            float smooth_scale = (2.0f * lambda_ts) / ((float)(B - 1) * (float)out_dim);
            for (int i = 0; i < B - 1; ++i) {
                const float* p_next = pred + (i + 1) * out_dim;
                const float* p_curr = pred + i * out_dim;
                float* dz_next = dz_last + (i + 1) * out_dim;
                float* dz_curr = dz_last + i * out_dim;
                for (int j = 0; j < out_dim; ++j) {
                    float s_diff = (p_next[j] - p_curr[j]) * smooth_scale;
                    dz_next[j] += s_diff;
                    dz_curr[j] -= s_diff;
                }
            }
        }

        for (int l = n_layers - 1; l >= 0; --l) {
            int d_in = dims[l];
            int d_out = dims[l + 1];
            const float* prev_A = (l == 0) ? X_b : A[l - 1];
            const float* dz_l = dZ[l];
            float* dw_l = dW[l];
            float* db_l = db[l];

            #pragma omp parallel for schedule(static)
            for (int k = 0; k < d_in; ++k) {
                float* dw_row = dw_l + k * d_out;
                for (int j = 0; j < d_out; ++j) {
                    dw_row[j] = 0.0f;
                }
                for (int i = 0; i < B; ++i) {
                    float a_ik = prev_A[i * d_in + k];
                    const float* dz_row = dz_l + i * d_out;
                    #pragma omp simd
                    for (int j = 0; j < d_out; ++j) {
                        dw_row[j] += a_ik * dz_row[j];
                    }
                }
            }

            for (int j = 0; j < d_out; ++j) {
                db_l[j] = 0.0f;
            }
            for (int i = 0; i < B; ++i) {
                const float* dz_row = dz_l + i * d_out;
                for (int j = 0; j < d_out; ++j) {
                    db_l[j] += dz_row[j];
                }
            }

            if (l > 0) {
                const float* W_l = W[l];
                const float* a_prev = A[l - 1];
                float* dz_prev = dZ[l - 1];

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < B; ++i) {
                    const float* dz_row = dz_l + i * d_out;
                    const float* a_row = a_prev + i * d_in;
                    float* dz_p_row = dz_prev + i * d_in;
                    for (int k = 0; k < d_in; ++k) {
                        const float* w_row = W_l + k * d_out;
                        float sum = 0.0f;
                        #pragma omp simd reduction(+:sum)
                        for (int j = 0; j < d_out; ++j) {
                            sum += dz_row[j] * w_row[j];
                        }
                        float a_val = a_row[k];
                        dz_p_row[k] = sum * (1.0f - a_val * a_val);
                    }
                }
            }
        }

        (*step_counter) += 1;
        int t = *step_counter;
        float bc1 = 1.0f - powf(beta1, (float)t);
        float bc2 = 1.0f - powf(beta2, (float)t);
        float step_size = lr * sqrtf(bc2) / bc1;

        for (int l = 0; l < n_layers; ++l) {
            int n_w = dims[l] * dims[l + 1];
            int n_b = dims[l + 1];
            float* w_arr = W[l];
            float* dw_arr = dW[l];
            float* mw_arr = mW[l];
            float* vw_arr = vW[l];

            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < n_w; ++idx) {
                float g = dw_arr[idx] + weight_decay * w_arr[idx];
                float m = beta1 * mw_arr[idx] + (1.0f - beta1) * g;
                float v = beta2 * vw_arr[idx] + (1.0f - beta2) * g * g;
                mw_arr[idx] = m;
                vw_arr[idx] = v;
                w_arr[idx] -= step_size * m / (sqrtf(v) + eps);
            }

            float* b_arr = b[l];
            float* db_arr = db[l];
            float* mb_arr = mb[l];
            float* vb_arr = vb[l];
            for (int j = 0; j < n_b; ++j) {
                float g = db_arr[j];
                float m = beta1 * mb_arr[j] + (1.0f - beta1) * g;
                float v = beta2 * vb_arr[j] + (1.0f - beta2) * g * g;
                mb_arr[j] = m;
                vb_arr[j] = v;
                b_arr[j] -= step_size * m / (sqrtf(v) + eps);
            }
        }
    }

    for (int l = 0; l < n_layers; ++l) {
        free(A[l]);
        free(dZ[l]);
        free(dW[l]);
        free(db[l]);
    }
    free(A);
    free(dZ);
    free(dW);
    free(db);

    return (float)(epoch_mse_sum / (double)N);
}
"""


def _get_c_engine() -> ctypes.CDLL:
    """Compile (if needed) and load the native AVX2+OpenMP MLP training library."""
    so_path = Path('/tmp/m300_mlp_engine_v1.so')
    c_path = Path('/tmp/m300_mlp_engine_v1.c')
    if not so_path.exists():
        c_path.write_text(_C_ENGINE_SOURCE, encoding='utf-8')
        subprocess.check_call(
            [
                'gcc',
                '-O3',
                '-march=native',
                '-ffast-math',
                '-fopenmp',
                '-shared',
                '-fPIC',
                str(c_path),
                '-o',
                str(so_path),
                '-lm',
            ]
        )
    lib = ctypes.CDLL(str(so_path))
    lib.eval_dataset_mse.restype = ctypes.c_float
    lib.train_epoch_caps.restype = ctypes.c_float
    return lib


class CleanDataset:
    """Sliding-window dataset built from synthetic or flight logs with residual labels.

    Attributes:
        h (int): Temporal window length [samples at the log rate].
        feature_cols (list[str]): Input columns per time step.
        input_mean (np.ndarray): Per-feature mean used for normalization.
        input_std (np.ndarray): Per-feature standard deviation used for normalization.
        target_mean (np.ndarray): Per-axis label mean.
        target_std (np.ndarray): Per-axis label standard deviation.
        X (np.ndarray): Normalized flattened windows of shape ``(N, n_feat * h)``.
        Y (np.ndarray): Normalized labels of shape ``(N, 6)``.
        file_slices (list[tuple[str, int, int]]): Start/end row indices for each flight.
    """

    def __init__(
        self,
        files: list[Path],
        feature_cols: list[str] | None = None,
        h: int = 1,
        stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.h = h
        self.feature_cols = feature_cols or FEATURE_SETS['wind']
        per_file: list[tuple[str, np.ndarray, np.ndarray]] = []
        for path in sorted(files):
            x_in, y_tg = self._read(path, self.feature_cols)
            if len(x_in) >= h:
                per_file.append((path.stem, x_in, y_tg))
        if not per_file:
            raise ValueError(f'No usable flight logs among {len(files)} files.')

        if stats is None:
            raw_x = np.vstack([x for _, x, _ in per_file])
            raw_y = np.vstack([y for _, _, y in per_file])
            self.input_mean = raw_x.mean(axis=0).astype(np.float32)
            self.input_std = np.maximum(raw_x.std(axis=0), 1e-3).astype(np.float32)
            self.target_mean = raw_y.mean(axis=0).astype(np.float32)
            self.target_std = np.maximum(raw_y.std(axis=0), 1e-5).astype(np.float32)
        else:
            self.input_mean, self.input_std, self.target_mean, self.target_std = stats

        x_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        self.file_slices: list[tuple[str, int, int]] = []
        cursor = 0
        for stem, x_in, y_tg in per_file:
            norm_x = ((x_in - self.input_mean) / self.input_std).astype(np.float32)
            norm_y = ((y_tg - self.target_mean) / self.target_std).astype(np.float32)
            win = sliding_window_view(norm_x, (h, norm_x.shape[1]))[:, 0, :, :]
            flat_win = win.reshape(len(win), -1)
            x_blocks.append(flat_win)
            y_blocks.append(norm_y[h - 1 :])
            n_seq = len(flat_win)
            self.file_slices.append((stem, cursor, cursor + n_seq))
            cursor += n_seq

        self.X = np.ascontiguousarray(np.vstack(x_blocks), dtype=np.float32)
        self.Y = np.ascontiguousarray(np.vstack(y_blocks), dtype=np.float32)

    @property
    def stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(input_mean, input_std, target_mean, target_std)``."""
        return self.input_mean, self.input_std, self.target_mean, self.target_std

    def raw_last_step(self, x_norm: np.ndarray | None = None) -> np.ndarray:
        """Return the physical-unit features of the newest step in each window."""
        x = self.X if x_norm is None else x_norm
        n_feat = len(self.feature_cols)
        return x[:, -n_feat:] * self.input_std + self.input_mean

    def labels_phys(self) -> np.ndarray:
        """Return the labels in physical units [m/s²] and [rad/s²] (body frame)."""
        return self.Y * self.target_std + self.target_mean

    def noise_std_norm(self, scale: float) -> np.ndarray:
        """Return the per-column input noise std in normalized units, tiled over ``h``."""
        sigma = np.array(
            [SENSOR_NOISE_STD[c] for c in self.feature_cols], dtype=np.float32
        )
        return np.tile(scale * sigma / self.input_std, self.h).astype(np.float32)

    @staticmethod
    def _read(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Load airborne rows of the feature and label columns from a flight log CSV."""
        with open(path, 'r', encoding='utf-8', newline='') as f:
            header = next(csv.reader(f))
        data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        col = {name: i for i, name in enumerate(header)}
        # Ground segments before takeoff and after landing carry no residual
        # information and would dominate the hover statistics.
        airborne = data[:, col['Pos_Alt_m']] > 0.05
        x_in = data[airborne][:, [col[c] for c in feature_cols]]
        y_tg = data[airborne][:, [col[c] for c in LABEL_COLS]]
        return x_in.astype(np.float32), y_tg.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)


DroneDataset = CleanDataset


class DroneResidualMLP:
    """Configurable Multi-Layer Perceptron for 6-DoF residual acceleration prediction.

    Supports arbitrary hidden layer depths and widths with ``tanh`` hidden
    activations and a linear output head. Stores weights as contiguous float32
    NumPy arrays compatible with both the native C/OpenMP engine and real-time
    ROS 2 inference.

    Attributes:
        h (int): Temporal window length [samples].
        input_dim (int): Number of features per time step.
        target_dim (int): Number of predicted residual accelerations (6).
        hidden_dims (list[int]): Neurons in each hidden layer.
        dims (list[int]): Full layer dimension list ``[input_dim * h, *hidden_dims, target_dim]``.
        weights (list[np.ndarray]): Weight matrices ``W_l`` for each linear layer.
        biases (list[np.ndarray]): Bias vectors ``b_l`` for each linear layer.
    """

    def __init__(
        self,
        h: int = 1,
        input_dim: int = 13,
        target_dim: int = 6,
        hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        seed: int = 42,
    ) -> None:
        self.h = h
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dims = [input_dim * h, *self.hidden_dims, target_dim]

        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        # Xavier/Glorot uniform initialization for tanh layers
        for l_idx in range(len(self.dims) - 1):
            d_in = self.dims[l_idx]
            d_out = self.dims[l_idx + 1]
            limit = math.sqrt(6.0 / (d_in + d_out))
            w = rng.uniform(-limit, limit, size=(d_in, d_out)).astype(np.float32)
            b = np.zeros(d_out, dtype=np.float32)
            self.weights.append(np.ascontiguousarray(w))
            self.biases.append(np.ascontiguousarray(b))

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable scalar weights and biases."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass predictions for input batch ``x``."""
        act = np.ascontiguousarray(x, dtype=np.float32)
        if act.ndim == 1:
            for l_idx in range(len(self.weights) - 1):
                act = np.tanh(act @ self.weights[l_idx] + self.biases[l_idx])
            return act @ self.weights[-1] + self.biases[-1]

        lib = _get_c_engine()
        n_samples = len(act)
        n_layers = len(self.weights)
        dims_arr = (ctypes.c_int * len(self.dims))(*self.dims)
        w_ptrs = (ctypes.c_void_p * n_layers)(
            *[w.ctypes.data for w in self.weights]
        )
        b_ptrs = (ctypes.c_void_p * n_layers)(
            *[b.ctypes.data for b in self.biases]
        )
        y_dummy = np.zeros((n_samples, self.target_dim), dtype=np.float32)
        y_pred = np.empty((n_samples, self.target_dim), dtype=np.float32)
        lib.eval_dataset_mse(
            n_samples,
            4096,
            n_layers,
            dims_arr,
            act.ctypes.data_as(ctypes.c_void_p),
            y_dummy.ctypes.data_as(ctypes.c_void_p),
            w_ptrs,
            b_ptrs,
            y_pred.ctypes.data_as(ctypes.c_void_p),
        )
        return y_pred

    def save(
        self, filepath: Path, extra: dict[str, np.ndarray] | None = None
    ) -> None:
        """Save model architecture, weights, and optional metadata to a ``.npz`` archive."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            'h': np.array(self.h, dtype=np.int32),
            'input_dim': np.array(self.input_dim, dtype=np.int32),
            'target_dim': np.array(self.target_dim, dtype=np.int32),
            'hidden_dims': np.array(self.hidden_dims, dtype=np.int32),
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            payload[f'W_{i}'] = w
            payload[f'b_{i}'] = b
        if extra:
            payload.update(extra)
        np.savez(filepath, **payload)

    @classmethod
    def load(cls, filepath: Path) -> DroneResidualMLP:
        """Load a trained ``DroneResidualMLP`` from a ``.npz`` checkpoint."""
        ckpt = np.load(filepath)
        h = int(ckpt['h'])
        input_dim = int(ckpt['input_dim'])
        target_dim = int(ckpt['target_dim'])
        hidden_dims = [int(x) for x in ckpt['hidden_dims']]
        model = cls(
            h=h,
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dims=hidden_dims,
        )
        for i in range(len(model.weights)):
            model.weights[i] = np.ascontiguousarray(ckpt[f'W_{i}'], dtype=np.float32)
            model.biases[i] = np.ascontiguousarray(ckpt[f'b_{i}'], dtype=np.float32)
        return model


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calculate physical-unit regression and smoothness metrics across all 6 axes.

    Args:
        y_true: Ground-truth residual matrix of shape ``(N, 6)``.
        y_pred: Predicted residual matrix of shape ``(N, 6)``.

    Returns:
        Dictionary with overall and per-axis RMSE, MAE, R², and temporal jerk.
    """
    err = y_pred - y_true
    mse_per_axis = np.mean(err**2, axis=0)
    rmse_per_axis = np.sqrt(mse_per_axis)
    mae_per_axis = np.mean(np.abs(err), axis=0)

    ss_res = np.sum(err**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0) + 1e-12
    r2_per_axis = 1.0 - (ss_res / ss_tot)

    pred_diff = np.diff(y_pred, axis=0)
    smoothness_rms = float(np.sqrt(np.mean(pred_diff**2)))

    return {
        'overall_mse': float(np.mean(mse_per_axis)),
        'overall_rmse': float(np.sqrt(np.mean(mse_per_axis))),
        'overall_mae': float(np.mean(mae_per_axis)),
        'overall_r2': float(np.mean(r2_per_axis)),
        'linear_rmse_ms2': float(np.sqrt(np.mean(mse_per_axis[0:3]))),
        'angular_rmse_rads2': float(np.sqrt(np.mean(mse_per_axis[3:6]))),
        'linear_r2': float(np.mean(r2_per_axis[0:3])),
        'angular_r2': float(np.mean(r2_per_axis[3:6])),
        'smoothness_rms': smoothness_rms,
        'per_axis_rmse': {
            col: float(rmse_per_axis[i]) for i, col in enumerate(TARGET_COLS)
        },
        'per_axis_mae': {
            col: float(mae_per_axis[i]) for i, col in enumerate(TARGET_COLS)
        },
        'per_axis_r2': {
            col: float(r2_per_axis[i]) for i, col in enumerate(TARGET_COLS)
        },
    }


def train_single_architecture(
    arch_name: str,
    hidden_dims: list[int],
    train_ds: CleanDataset,
    val_ds: CleanDataset,
    test_ds: CleanDataset,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    lambda_ts: float = 0.0,
    seed: int = 42,
    models_dir: Path | None = None,
    input_dim: int | None = None,
    hover_anchor: bool = False,
    extra_meta: dict[str, np.ndarray] | None = None,
    epoch_hook: Callable[[int], None] | None = None,
    patience: int | None = 30,
    lr_floor: float = 0.05,
) -> dict[str, Any]:
    """Train one MLP candidate architecture, save its best weights, and evaluate it.

    Args:
        arch_name: Identifier for the architecture (e.g. ``'MLP_2L_64_32'``).
        hidden_dims: List of neuron counts for each hidden layer.
        train_ds: Normalized training dataset.
        val_ds: Normalized validation dataset.
        test_ds: Normalized unseen test dataset.
        epochs: Number of training epochs.
        batch_size: Contiguous mini-batch size.
        lr: Initial Adam learning rate.
        weight_decay: L2 weight decay coefficient.
        lambda_ts: Weight for CAPS temporal smoothness loss term.
        seed: Random seed for reproducible weight initialization.
        models_dir: Root folder for checkpoints (defaults to ``paths.NN_MODELS_DIR``).
        input_dim: Features per time step.
        hover_anchor: Re-center the output bias at ``X = 0`` after every epoch.
        extra_meta: Additional arrays stored inside every saved checkpoint.
        epoch_hook: Called with the epoch index before each epoch; may shuffle or
            augment ``train_ds.X`` / ``train_ds.Y`` in place.
        patience: Stop after this many epochs without validation improvement.
        lr_floor: Final learning rate as a fraction of ``lr`` for the cosine schedule.

    Returns:
        Dictionary containing training curves, best epoch, and val/test metrics.
    """
    lib = _get_c_engine()
    model = DroneResidualMLP(
        h=train_ds.h,
        input_dim=input_dim or len(train_ds.feature_cols),
        target_dim=len(TARGET_COLS),
        hidden_dims=hidden_dims,
        seed=seed,
    )

    n_layers = len(model.weights)
    dims_arr = (ctypes.c_int * len(model.dims))(*model.dims)

    mw = [np.zeros_like(w) for w in model.weights]
    vw = [np.zeros_like(w) for w in model.weights]
    mb = [np.zeros_like(b) for b in model.biases]
    vb = [np.zeros_like(b) for b in model.biases]

    w_ptrs = (ctypes.c_void_p * n_layers)(*[w.ctypes.data for w in model.weights])
    b_ptrs = (ctypes.c_void_p * n_layers)(*[b.ctypes.data for b in model.biases])
    mw_ptrs = (ctypes.c_void_p * n_layers)(*[m.ctypes.data for m in mw])
    vw_ptrs = (ctypes.c_void_p * n_layers)(*[v.ctypes.data for v in vw])
    mb_ptrs = (ctypes.c_void_p * n_layers)(*[m.ctypes.data for m in mb])
    vb_ptrs = (ctypes.c_void_p * n_layers)(*[v.ctypes.data for v in vb])

    step_counter = ctypes.c_int(0)
    best_val_loss = float('inf')
    best_epoch = 0

    arch_dir = (models_dir or paths.NN_MODELS_DIR) / arch_name
    arch_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = arch_dir / 'best_model.npz'

    train_losses: list[float] = []
    val_losses: list[float] = []

    print(
        f'\n--- Training {arch_name} | Layers: {model.dims} | '
        f'Params: {model.num_parameters:,} ---'
    )
    t0 = time.perf_counter()

    for epoch in range(epochs):
        progress = epoch / max(1, epochs - 1)
        curr_lr = lr * (
            lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )

        if epoch_hook is not None:
            epoch_hook(epoch)

        train_mse = float(
            lib.train_epoch_caps(
                len(train_ds.X),
                batch_size,
                n_layers,
                dims_arr,
                train_ds.X.ctypes.data_as(ctypes.c_void_p),
                train_ds.Y.ctypes.data_as(ctypes.c_void_p),
                w_ptrs,
                b_ptrs,
                mw_ptrs,
                vw_ptrs,
                mb_ptrs,
                vb_ptrs,
                ctypes.byref(step_counter),
                ctypes.c_float(curr_lr),
                ctypes.c_float(weight_decay),
                ctypes.c_float(lambda_ts),
            )
        )

        if hover_anchor:
            hover_zero = np.zeros(model.dims[0], dtype=np.float32)
            hover_pred = model.forward(hover_zero)
            for ax_idx in (0, 1, 3, 4, 5):
                model.biases[-1][ax_idx] -= hover_pred[ax_idx]

        val_mse = float(
            lib.eval_dataset_mse(
                len(val_ds.X),
                4096,
                n_layers,
                dims_arr,
                val_ds.X.ctypes.data_as(ctypes.c_void_p),
                val_ds.Y.ctypes.data_as(ctypes.c_void_p),
                w_ptrs,
                b_ptrs,
                None,
            )
        )

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        saved_flag = ''
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch + 1
            model.save(best_ckpt_path, extra_meta)
            saved_flag = ' [BEST]'

        if (epoch + 1) % 10 == 0 or epoch == 0 or saved_flag:
            print(
                f'  Epoch {epoch + 1:02d}/{epochs:02d} | '
                f'LR: {curr_lr:.5f} | '
                f'Train MSE: {train_mse:.6f} | '
                f'Val MSE: {val_mse:.6f}{saved_flag}'
            )

        if patience is not None and (epoch + 1 - best_epoch) >= patience:
            print(
                f'  Early stop at epoch {epoch + 1} '
                f'(no val improvement for {patience}).'
            )
            break

    train_time_s = time.perf_counter() - t0

    best_model = DroneResidualMLP.load(best_ckpt_path)

    t_inf0 = time.perf_counter()
    y_val_pred_norm = best_model.forward(val_ds.X)
    y_test_pred_norm = best_model.forward(test_ds.X)
    inf_time_us = (
        (time.perf_counter() - t_inf0)
        * 1e6
        / max(1, len(val_ds.X) + len(test_ds.X))
    )

    y_val_true = val_ds.Y * val_ds.target_std + val_ds.target_mean
    y_val_pred = y_val_pred_norm * val_ds.target_std + val_ds.target_mean

    y_test_true = test_ds.Y * test_ds.target_std + test_ds.target_mean
    y_test_pred = y_test_pred_norm * test_ds.target_std + test_ds.target_mean

    val_metrics = compute_metrics(y_val_true, y_val_pred)
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    summary = {
        'name': arch_name,
        'hidden_dims': hidden_dims,
        'full_dims': best_model.dims,
        'num_parameters': best_model.num_parameters,
        'best_epoch': best_epoch,
        'best_val_norm_mse': best_val_loss,
        'train_time_s': round(train_time_s, 2),
        'inference_us_per_sample': round(inf_time_us, 3),
        'checkpoint_path': str(best_ckpt_path.relative_to(paths.PROJECT_ROOT)),
        'checkpoint_abs': str(best_ckpt_path),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }

    with open(arch_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(
        f'-> Finished {arch_name} in {train_time_s:.1f}s | '
        f'Best Epoch: {best_epoch} | '
        f'Val Norm MSE: {best_val_loss:.6f} | '
        f'Test R²: {test_metrics["overall_r2"]:.4f} | '
        f'Test Lin RMSE: {test_metrics["linear_rmse_ms2"]:.4f} m/s² | '
        f'Test Ang RMSE: {test_metrics["angular_rmse_rads2"]:.4f} rad/s²'
    )
    return summary


def make_epoch_hook(
    ds: CleanDataset,
    noise_scale: float,
    seed: int = 0,
) -> Callable[[int], None]:
    """Build a hook that reshuffles and re-noises the training set before every epoch."""
    rng = np.random.default_rng(seed)
    x_clean = ds.X.copy()
    y_clean = ds.Y.copy()
    sigma = ds.noise_std_norm(noise_scale) if noise_scale > 0.0 else None

    def hook(epoch: int) -> None:
        del epoch
        perm = rng.permutation(len(x_clean))
        if sigma is None:
            ds.X[:] = x_clean[perm]
        else:
            noise = rng.standard_normal(x_clean.shape, dtype=np.float32) * sigma
            ds.X[:] = x_clean[perm] + noise
        ds.Y[:] = y_clean[perm]

    return hook


def physics_regressors(raw: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """Build the linear-in-parameters regressor matrix of the analytic residual."""
    col = {c: i for i, c in enumerate(feature_cols)}
    p = raw[:, col['Rate_p_rads']]
    q = raw[:, col['Rate_q_rads']]
    r = raw[:, col['Rate_r_rads']]
    return np.column_stack(
        [np.ones(len(raw)), raw, p * q, q * r, r * p, p * p, r * r]
    ).astype(np.float64)


def fit_physics_baseline(
    train_ds: CleanDataset,
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Fit the linear physics baseline on clean train data and score it.

    Args:
        train_ds: Training dataset (clean inputs).
        eval_sets: Name -> ``(normalized X, physical labels)`` to evaluate on.

    Returns:
        Coefficient matrix ``(n_regressors, 6)`` and metrics per evaluation set.
    """
    phi = physics_regressors(train_ds.raw_last_step(), train_ds.feature_cols)
    coef, *_ = np.linalg.lstsq(
        phi, train_ds.labels_phys().astype(np.float64), rcond=None
    )
    metrics = {}
    for name, (x_norm, y_true) in eval_sets.items():
        phi_e = physics_regressors(
            train_ds.raw_last_step(x_norm), train_ds.feature_cols
        )
        metrics[name] = compute_metrics(y_true, (phi_e @ coef).astype(np.float32))
    return coef, metrics


def _split_files(base: Path, split: str, include_wind: bool) -> list[Path]:
    """List CSVs of a split, optionally dropping the windy ``*_w`` missions."""
    files = sorted((base / split).glob('*.csv'))
    if not include_wind:
        files = [f for f in files if not f.stem.endswith('_w')]
    return files


def _r2_line(metrics: dict[str, Any]) -> str:
    """Format per-axis R² compactly for console output."""
    return ' '.join(f'{v:.5f}' for v in metrics['per_axis_r2'].values())


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, train the candidate MLPs, and export the best one."""
    parser = argparse.ArgumentParser(
        description='Train residual MLPs on synthetic flight logs with analytic labels.'
    )
    parser.add_argument(
        '--domain', default='real', help="Flight log domain (default: 'real')."
    )
    parser.add_argument(
        '--features',
        default='wind',
        choices=sorted(FEATURE_SETS),
        help=(
            "'base' uses velocities, rates and rotor wrench (no-wind logs only); "
            "'wind' adds body-frame wind and trains on every log (default: wind)."
        ),
    )
    parser.add_argument(
        '--window', type=int, default=1, help='History length h (default: 1).'
    )
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument(
        '--patience',
        type=int,
        default=30,
        help='Early-stopping patience in epochs (default: 30).',
    )
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument(
        '--input-noise',
        type=float,
        default=0.0,
        help=(
            'Scale of the sensor-like input noise added to training inputs each epoch '
            '(0 = clean, 1 = nominal SENSOR_NOISE_STD). Labels stay exact.'
        ),
    )
    parser.add_argument(
        '--arch',
        nargs='+',
        choices=sorted(CLEAN_ARCHITECTURES),
        default=list(CLEAN_ARCHITECTURES),
        help='Subset of candidate architectures to train (default: all).',
    )
    parser.add_argument(
        '--activate',
        action='store_true',
        help='Copy the best model into data/nn_models so dynamic_node loads it.',
    )
    args = parser.parse_args(argv)

    base = paths.FLIGHT_LOGS_DIR / args.domain
    feature_cols = FEATURE_SETS[args.features]
    include_wind = args.features == 'wind'
    split_files = {
        s: _split_files(base, s, include_wind) for s in ('train', 'val', 'test')
    }
    if not all(split_files.values()):
        print(f'Flight logs missing in {base}. Run batch_sim.py first.')
        return 1

    train_ds = CleanDataset(split_files['train'], feature_cols, args.window)
    val_ds = CleanDataset(
        split_files['val'], feature_cols, args.window, train_ds.stats
    )
    test_ds = CleanDataset(
        split_files['test'], feature_cols, args.window, train_ds.stats
    )
    print(
        f'=== Residual MLP Training ({args.domain}, features={args.features}, '
        f'h={args.window}, input_noise={args.input_noise}) ===\n'
        f'Samples -> train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}'
    )

    noise_rng = np.random.default_rng(1234)
    x_test_noisy = test_ds.X + (
        noise_rng.standard_normal(test_ds.X.shape, dtype=np.float32)
        * test_ds.noise_std_norm(1.0)
    )
    y_test = test_ds.labels_phys()
    eval_sets = {'test': (test_ds.X, y_test), 'test_noisy': (x_test_noisy, y_test)}

    _, baseline = fit_physics_baseline(train_ds, eval_sets)
    print(
        f'Physics LSQ baseline | test R² {baseline["test"]["overall_r2"]:.6f} '
        f'| noisy-test R² {baseline["test_noisy"]["overall_r2"]:.5f}'
    )

    nominal_cfg, _ = load_aircraft_config('sim')
    suffix = f'_noise{args.input_noise:g}' if args.input_noise > 0.0 else ''
    out_dir = paths.NN_MODELS_DIR / f'{args.features}{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'feature_set': np.array(args.features),
        'input_mean': train_ds.input_mean,
        'input_std': train_ds.input_std,
        'target_mean': train_ds.target_mean,
        'target_std': train_ds.target_std,
        'rotor_params': rotor_params_from_dynamics(nominal_cfg.to_dynamics_array()),
    }

    results: list[dict[str, Any]] = []
    for name in args.arch:
        res = train_single_architecture(
            arch_name=name,
            hidden_dims=CLEAN_ARCHITECTURES[name],
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=0.0,
            lambda_ts=0.0,
            models_dir=out_dir,
            input_dim=len(feature_cols),
            hover_anchor=False,
            extra_meta=meta,
            epoch_hook=make_epoch_hook(train_ds, args.input_noise),
            patience=args.patience,
            lr_floor=0.05,
        )
        model = DroneResidualMLP.load(Path(res['checkpoint_abs']))
        y_pred = (
            model.forward(x_test_noisy) * test_ds.target_std + test_ds.target_mean
        )
        res['test_noisy_metrics'] = compute_metrics(y_test, y_pred)
        results.append(res)

    results.sort(key=lambda r: r['best_val_norm_mse'])
    best = results[0]
    best_path = out_dir / 'best_m350_model.npz'
    shutil.copyfile(best['checkpoint_abs'], best_path)
    for key in ('input_mean', 'input_std', 'target_mean', 'target_std'):
        np.save(out_dir / f'{key}.npy', meta[key])

    summary_payload = {
        'domain': args.domain,
        'feature_set': args.features,
        'feature_columns': feature_cols,
        'target_columns': LABEL_COLS,
        'window_h': args.window,
        'input_noise_scale': args.input_noise,
        'sensor_noise_std': {c: SENSOR_NOISE_STD[c] for c in feature_cols},
        'physics_baseline': baseline,
        'best_architecture': best['name'],
        'models': results,
    }
    with open(out_dir / 'models_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=4)

    print(
        '\nRank  Model               Params  Epoch   Val MSE   Test R²  '
        'Noisy R²  Lin RMSE  Ang RMSE'
    )
    for rank, r in enumerate(results, start=1):
        tm, nm = r['test_metrics'], r['test_noisy_metrics']
        print(
            f'{rank:<5} {r["name"]:<18} {r["num_parameters"]:>7,} {r["best_epoch"]:>6} '
            f'{r["best_val_norm_mse"]:>9.2e} {tm["overall_r2"]:>8.5f} {nm["overall_r2"]:>8.5f} '
            f'{tm["linear_rmse_ms2"]:>9.2e} {tm["angular_rmse_rads2"]:>9.2e}'
        )
    bt, bn = baseline['test'], baseline['test_noisy']
    print(
        f'{"-":<5} {"Physics LSQ":<18} {"-":>7} {"-":>6} {"-":>9} '
        f'{bt["overall_r2"]:>8.5f} {bn["overall_r2"]:>8.5f} '
        f'{bt["linear_rmse_ms2"]:>9.2e} {bt["angular_rmse_rads2"]:>9.2e}'
    )
    print(
        f'\nPer-axis test R² (u v w p q r)\n'
        f'  {best["name"]:<14} {_r2_line(best["test_metrics"])}'
    )
    print(f'  {"Physics LSQ":<14} {_r2_line(bt)}')
    print(
        f'Per-axis noisy-test R²\n'
        f'  {best["name"]:<14} {_r2_line(best["test_noisy_metrics"])}'
    )
    print(f'  {"Physics LSQ":<14} {_r2_line(bn)}')

    if args.activate:
        for name in (
            'best_m350_model.npz',
            'input_mean.npy',
            'input_std.npy',
            'target_mean.npy',
            'target_std.npy',
            'models_summary.json',
        ):
            shutil.copyfile(out_dir / name, paths.NN_MODELS_DIR / name)
        print(f'Activated {best["name"]} in {paths.NN_MODELS_DIR}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
