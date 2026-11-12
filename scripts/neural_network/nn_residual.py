"""Real-time 6-DoF residual acceleration predictor using the trained MLP.

Loads the best trained Multi-Layer Perceptron (``.npz`` weights and
normalization statistics from ``data/nn_models``) and maintains a 100 Hz
sliding window buffer ($h = 10$) with cascaded input wrench low-pass filtering,
zero-equilibrium hover anchoring, smooth deadzone gating, passivity damping,
and output low-pass smoothing to infer translational and rotational residual
accelerations during the 1000 Hz physics loop without inducing attitude
chattering or hover drift.
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

_CURRENT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_network.nn_features import build_features, FEATURE_SETS  # noqa: E402
import paths  # noqa: E402


class ResidualNNPredictor:
    """Real-time sliding-window MLP inference engine for 6-DoF dynamics compensation.

    Attributes:
        enabled (bool): Whether neural network residual compensation is active.
        ready (bool): Whether model weights and normalization files loaded cleanly.
        h (int): Temporal window length [samples at 100 Hz].
        hidden_dims (list[int]): Hidden layer dimensions of the loaded MLP.
        weights (list[np.ndarray]): MLP layer weight matrices.
        biases (list[np.ndarray]): MLP layer bias vectors.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        enabled: bool = False,
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.ready: bool = False
        self.h: int = 10
        self.hidden_dims: list[int] = []
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.input_mean = np.zeros(16, dtype=np.float32)
        self.input_std = np.ones(16, dtype=np.float32)
        self.target_mean = np.zeros(6, dtype=np.float32)
        self.target_std = np.ones(6, dtype=np.float32)
        self._model_dir: Path = model_dir or paths.NN_MODELS_DIR
        self._loaded_mtime: float = 0.0
        self._hover_bias = np.zeros(6, dtype=np.float64)
        self._history: deque[np.ndarray] = deque(maxlen=self.h)
        self._u_stage1: np.ndarray | None = None
        self._u_filtered: np.ndarray | None = None
        self._prev_omega = np.zeros(3, dtype=np.float64)
        self._filtered_prediction = np.zeros(6, dtype=np.float64)
        self._last_prediction = np.zeros(6, dtype=np.float64)
        self.feature_set: str = 'legacy'
        self._n_features: int = 16
        self._rotor_params = np.zeros(5, dtype=np.float64)

        self.load_artifacts(self._model_dir)

    def _raw_forward(self, window_vec: np.ndarray) -> np.ndarray:
        """Evaluate the MLP forward pass and unnormalize to physical units."""
        act = window_vec
        for l_idx in range(len(self.weights) - 1):
            act = np.tanh(act @ self.weights[l_idx] + self.biases[l_idx])
        pred_norm = act @ self.weights[-1] + self.biases[-1]
        return ((pred_norm * self.target_std) + self.target_mean).astype(np.float64)

    def load_artifacts(self, model_dir: Path | None = None) -> bool:
        """Load trained MLP weights and normalization arrays from ``model_dir``.

        Args:
            model_dir: Directory containing ``best_m350_model.npz`` and ``*.npy`` stats.

        Returns:
            ``True`` if all model artifacts were loaded successfully.
        """
        target_dir = model_dir or self._model_dir
        self._model_dir = target_dir

        ckpt_path = target_dir / 'best_m350_model.npz'
        in_mean_path = target_dir / 'input_mean.npy'
        in_std_path = target_dir / 'input_std.npy'
        tg_mean_path = target_dir / 'target_mean.npy'
        tg_std_path = target_dir / 'target_std.npy'

        artifact_files = (
            ckpt_path,
            in_mean_path,
            in_std_path,
            tg_mean_path,
            tg_std_path,
        )
        if not all(p.exists() for p in artifact_files):
            self.ready = False
            return False

        ckpt = np.load(ckpt_path)
        self.h = int(ckpt['h'])
        self.hidden_dims = [int(x) for x in ckpt['hidden_dims']]
        n_layers = len(self.hidden_dims) + 1

        self.weights = [
            np.ascontiguousarray(ckpt[f'W_{i}'], dtype=np.float32)
            for i in range(n_layers)
        ]
        self.biases = [
            np.ascontiguousarray(ckpt[f'b_{i}'], dtype=np.float32)
            for i in range(n_layers)
        ]

        self.input_mean = np.load(in_mean_path).astype(np.float32)
        self.input_std = np.load(in_std_path).astype(np.float32)
        self.target_mean = np.load(tg_mean_path).astype(np.float32)
        self.target_std = np.load(tg_std_path).astype(np.float32)
        self._loaded_mtime = max(p.stat().st_mtime for p in artifact_files)

        self._history = deque(maxlen=self.h)
        self._u_stage1 = None
        self._u_filtered = None
        self._prev_omega = np.zeros(3, dtype=np.float64)
        self._filtered_prediction = np.zeros(6, dtype=np.float64)
        self._last_prediction = np.zeros(6, dtype=np.float64)

        # Clean checkpoints carry their own feature definition and statistics, so the
        # runtime can never drift from what batch_sim.py --clean logged.
        if 'feature_set' in ckpt.files:
            self.feature_set = str(ckpt['feature_set'])
            self._n_features = len(FEATURE_SETS[self.feature_set])
            self._rotor_params = np.asarray(ckpt['rotor_params'], dtype=np.float64)
            self.input_mean = np.asarray(ckpt['input_mean'], dtype=np.float32)
            self.input_std = np.asarray(ckpt['input_std'], dtype=np.float32)
            self.target_mean = np.asarray(ckpt['target_mean'], dtype=np.float32)
            self.target_std = np.asarray(ckpt['target_std'], dtype=np.float32)
            self._hover_bias = np.zeros(6, dtype=np.float64)
            self.ready = True
            return True
        self.feature_set = 'legacy'

        # Evaluate network at exact stationary hover equilibrium (all normalized features = 0)
        # to anchor horizontal and rotational residuals strictly at 0.0 when at rest.
        hover_feat = np.array(
            [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                float(self.input_mean[12]), 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        hover_norm = (hover_feat - self.input_mean) / self.input_std
        hover_window = np.tile(hover_norm, self.h)
        hover_out = self._raw_forward(hover_window)
        self._hover_bias = np.array(
            [hover_out[0], hover_out[1], 0.0, hover_out[3], hover_out[4], hover_out[5]],
            dtype=np.float64,
        )
        self.ready = True
        return True

    def reload_if_updated(self) -> bool:
        """Reload model weights from disk if any artifact file was modified.

        Returns:
            ``True`` if newer weights were detected and reloaded.
        """
        ckpt_path = self._model_dir / 'best_m350_model.npz'
        if not ckpt_path.exists():
            return False
        try:
            curr_mtime = ckpt_path.stat().st_mtime
            if curr_mtime > self._loaded_mtime + 1e-3 or not self.ready:
                return self.load_artifacts(self._model_dir)
        except OSError:
            pass
        return False

    def reset(self) -> None:
        """Clear the temporal history buffer and zero out cached predictions."""
        self._history.clear()
        self._u_stage1 = None
        self._u_filtered = None
        self._prev_omega = np.zeros(3, dtype=np.float64)
        self._filtered_prediction = np.zeros(6, dtype=np.float64)
        self._last_prediction = np.zeros(6, dtype=np.float64)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable neural network residual output.

        Args:
            enabled: ``True`` to inject predicted residuals, ``False`` to output zeros.
        """
        if enabled:
            self.reload_if_updated()
        self.enabled = bool(enabled)
        if not self.enabled:
            self._filtered_prediction = np.zeros(6, dtype=np.float64)
            self._last_prediction = np.zeros(6, dtype=np.float64)

    def step(
        self,
        state: np.ndarray,
        u_virtual: np.ndarray,
        wind_ned: np.ndarray | None = None,
    ) -> np.ndarray:
        """Record a 100 Hz telemetry sample and return the 6-DoF residual prediction.

        Even when ``self.enabled`` is ``False``, the rolling buffer is updated so
        that enabling the network mid-flight immediately has a valid window.

        Args:
            state: Current 20-element state vector of the quadcopter.
            u_virtual: Current 4-element control wrench ``[Fz, Tx, Ty, Tz]``
                (only used by legacy checkpoints).
            wind_ned: Current wind velocity [m/s] (NED frame), used by clean
                checkpoints trained with the ``'wind'`` feature set.

        Returns:
            Array of 6 residual accelerations
            ``[Delta_a_u, Delta_a_v, Delta_a_w, Delta_alpha_p, Delta_alpha_q, Delta_alpha_r]``
            in body-frame physical units (``[m/s²]`` and ``[rad/s²]``).
        """
        if not self.ready:
            return np.zeros(6, dtype=np.float64)
        if self.feature_set != 'legacy':
            return self._step_clean(state, wind_ned)

        # Second-order cascaded low-pass filter on the virtual control wrench (~3 Hz cutoff
        # at 100 Hz) to break any algebraic feedback loop between the 1000 Hz rate-loop
        # derivative kick -(omega - prev_omega)/dt, ROS 2 DDS jitter, and MLP torque inputs.
        u_curr = np.asarray(u_virtual, dtype=np.float64)
        if self._u_stage1 is None or self._u_filtered is None:
            self._u_stage1 = u_curr.copy()
            self._u_filtered = u_curr.copy()
        else:
            alpha_u = 0.22
            self._u_stage1 = alpha_u * u_curr + (1.0 - alpha_u) * self._u_stage1
            self._u_filtered = (
                alpha_u * self._u_stage1 + (1.0 - alpha_u) * self._u_filtered
            )

        rot = Rotation.from_euler('xyz', state[6:9])
        roll, pitch, yaw = rot.as_euler('xyz')

        raw_feat = np.array(
            [
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
                state[5],
                math.sin(roll),
                math.cos(roll),
                math.sin(pitch),
                math.cos(pitch),
                math.sin(yaw),
                math.cos(yaw),
                self._u_filtered[0],
                self._u_filtered[1],
                self._u_filtered[2],
                self._u_filtered[3],
            ],
            dtype=np.float32,
        )

        # Soft-clip normalized features to [-3.5, +3.5] so aggressive manual inputs
        # never push hidden neurons into out-of-distribution saturation.
        norm_feat = (raw_feat - self.input_mean) / self.input_std
        norm_feat = np.float32(3.5) * np.tanh(norm_feat / np.float32(3.5))
        self._history.append(norm_feat)

        if not self.enabled or len(self._history) < self.h:
            self._prev_omega = state[3:6].copy()
            self._filtered_prediction = np.zeros(6, dtype=np.float64)
            self._last_prediction = np.zeros(6, dtype=np.float64)
            return self._last_prediction.copy()

        window_vec = np.concatenate(self._history)
        pred_phys = self._raw_forward(window_vec) - self._hover_bias

        # Smooth deadzone activity gates: guarantee strictly zero horizontal and rotational
        # residual output when the drone is on the ground or hovering/settling at any altitude.
        vel_norm = math.sqrt(
            float(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        )
        omega_norm = math.sqrt(
            float(state[3] ** 2 + state[4] ** 2 + state[5] ** 2)
        )
        tau_norm = math.sqrt(
            float(
                self._u_filtered[1] ** 2
                + self._u_filtered[2] ** 2
                + self._u_filtered[3] ** 2
            )
        )
        alt_m = float(-state[11])

        ground_gate = math.tanh(max(0.0, (alt_m - 0.05) / 0.15) ** 2)
        vel_eff = max(0.0, vel_norm - 0.12)
        omega_eff = max(0.0, omega_norm - 0.05)
        tau_eff = max(0.0, tau_norm - 0.08)

        lin_gate = ground_gate * math.tanh((vel_eff / 0.30) ** 2)
        rot_gate = ground_gate * math.tanh(
            (omega_eff / 0.12) ** 2 + (tau_eff / 0.20) ** 2
        )

        # Passivity / anti-chatter damping on roll and pitch: if angular rate is
        # rapidly reversing sign, attenuate rotational residual injection.
        d_omega_rp = math.hypot(
            float(state[3] - self._prev_omega[0]),
            float(state[4] - self._prev_omega[1]),
        )
        self._prev_omega = state[3:6].copy()
        chatter_damp = 1.0 / (1.0 + (d_omega_rp / 0.08) ** 2)

        pred_phys[0] *= lin_gate
        pred_phys[1] *= lin_gate
        pred_phys[2] *= ground_gate
        pred_phys[3] *= rot_gate * chatter_damp
        pred_phys[4] *= rot_gate * chatter_damp
        pred_phys[5] *= rot_gate

        # Smooth physical envelope saturation matching the 99th-percentile structural bounds
        limits = np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.70], dtype=np.float64)
        pred_phys = limits * np.tanh(pred_phys / limits)

        # Output low-pass smoothing (cutoff ~3 Hz at 100 Hz) for bumpless transitions
        alpha_out = 0.18
        self._filtered_prediction = (
            alpha_out * pred_phys + (1.0 - alpha_out) * self._filtered_prediction
        )
        self._last_prediction = np.where(
            np.abs(self._filtered_prediction) < 1e-5,
            0.0,
            self._filtered_prediction,
        )
        return self._last_prediction.copy()

    def _step_clean(self, state: np.ndarray, wind_ned: np.ndarray | None) -> np.ndarray:
        """Run a clean checkpoint: plain forward pass, no gates or output filters.

        Clean models see the rotor-speed wrench instead of ``u_virtual``, which the
        motor dynamics already low-pass, so the closed loop stays smooth without the
        legacy deadzones, chatter damping, and EMA stages.
        """
        wind = np.zeros(3) if wind_ned is None else np.asarray(wind_ned, dtype=np.float64)
        feat = build_features(state, wind, self._rotor_params)[: self._n_features]
        norm = (feat.astype(np.float32) - self.input_mean) / self.input_std
        # Hard clip keeps extreme manual inputs inside the trained envelope.
        self._history.append(np.clip(norm, -6.0, 6.0).astype(np.float32))

        if not self.enabled or len(self._history) < self.h:
            self._last_prediction = np.zeros(6, dtype=np.float64)
            return self._last_prediction.copy()

        pred = self._raw_forward(np.concatenate(self._history))
        # Training data never contains ground contact, so fade out below 5 cm.
        alt_m = float(-state[11])
        pred *= min(1.0, max(0.0, (alt_m - 0.02) / 0.03))
        self._last_prediction = pred
        return pred.copy()

    @property
    def current_residual(self) -> np.ndarray:
        """Return the latest 6-DoF residual acceleration vector."""
        if not self.enabled or not self.ready:
            return np.zeros(6, dtype=np.float64)
        return self._last_prediction.copy()
