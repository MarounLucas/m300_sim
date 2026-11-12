"""Numerical integration solvers for state-space differential equations."""

from __future__ import annotations

import numpy as np


def forward_euler(x_current: np.ndarray, dx: np.ndarray, dt: float) -> np.ndarray:
    """Advance the state vector by one step using the Forward Euler method.

    Args:
        x_current: State vector at the current time step ``t``.
        dx: State derivative vector computed by the dynamics model.
        dt: Integration time step [s].

    Returns:
        Updated state vector at ``t + dt``.
    """
    return x_current + dt * dx
