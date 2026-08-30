"""Numerical integration methods for differential equations.

This module provides mathematical solvers to advance the UAV's state vector 
over time based on its computed derivatives.

Author: Lucas Maroun de Almeida
"""

import numpy as np


def forward_euler(x_current: np.ndarray, dx: np.ndarray, dt: float) -> np.ndarray:
    """Performs a single numerical integration step using the Forward Euler method.

    Computes the next state of a discrete-time system by projecting the current 
    state linearly along its derivative for a duration of dt.

    Args:
        x_current (np.ndarray): The state vector at the current time step (t).
        dx (np.ndarray): The state derivative vector computed by the dynamics.
        dt (float): The time integration step in seconds.

    Returns:
        np.ndarray: The updated state vector for the next time step (t + dt).
    """
    return x_current + dt * dx