"""Aerodynamic wind profile manager wrapping RotorPy disturbance models."""

from __future__ import annotations

from typing import Any

import numpy as np
from rotorpy.wind.default_winds import ConstantWind, NoWind, SinusoidWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP


class WindManager:
    """Manage constant and stochastic aerodynamic wind profiles in the NED frame.

    Convert spherical wind parameters (speed, heading, elevation, and gust
    intensity) into inertial NED components and query RotorPy wind models at
    each simulation step.
    """

    def __init__(
        self,
        wind_type: str = 'none',
        magnitude: float = 0.0,
        heading: float = 0.0,
        elevation: float = 0.0,
        gust_magnitude: float = 0.0,
    ) -> None:
        """Initialize the wind profile and pre-compute inertial NED components.

        Args:
            wind_type: Wind model identifier (``'none'``, ``'constant'``,
                ``'dryden'``, ``'dryden_lp'``, or ``'sinusoid'``).
            magnitude: Mean wind speed [m/s].
            heading: Horizontal wind direction angle in the NED plane [rad].
            elevation: Vertical wind elevation angle [rad].
            gust_magnitude: Peak gust speed for stochastic models [m/s].
        """
        self.wind_type = wind_type

        s_wh = np.sin(heading)
        c_wh = np.cos(heading)
        s_we = np.sin(elevation)
        c_we = np.cos(elevation)

        # Positive elevation points upward, which corresponds to -Z in NED.
        self.wx_ned = magnitude * c_wh * c_we
        self.wy_ned = magnitude * s_wh * c_we
        self.wz_ned = -magnitude * s_we

        # Approximate 1-sigma turbulence intensity as one third of the peak
        # excursion above the mean wind speed (3-sigma rule).
        sigma_total = (
            max(0.0, (gust_magnitude - magnitude) / 3.0)
            if gust_magnitude > magnitude
            else gust_magnitude
        )

        self.sig_wx = abs(sigma_total * c_wh * c_we)
        self.sig_wy = abs(sigma_total * s_wh * c_we)
        self.sig_wz = abs(sigma_total * s_we)

        self.wind_profile = self._build_profile()

    def _build_profile(self) -> Any:
        """Instantiate the RotorPy wind model matching ``self.wind_type``."""
        if self.wind_type == 'constant':
            return ConstantWind(wx=self.wx_ned, wy=self.wy_ned, wz=self.wz_ned)

        if self.wind_type == 'dryden':
            avg_wind = np.array([self.wx_ned, self.wy_ned, self.wz_ned])
            sig_wind = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            return DrydenGust(avg_wind=avg_wind, sig_wind=sig_wind, altitude=10.0)

        if self.wind_type == 'dryden_lp':
            avg_wind = np.array([self.wx_ned, self.wy_ned, self.wz_ned])
            sig_wind = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            return DrydenGustLP(
                avg_wind=avg_wind, sig_wind=sig_wind, altitude=10.0, tau=0.5
            )

        if self.wind_type == 'sinusoid':
            amplitudes = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            frequencies = np.array([0.5, 0.5, 0.5])
            return SinusoidWind(amplitudes=amplitudes, frequencies=frequencies)

        return NoWind()

    def get_wind(self, t: float, position: np.ndarray) -> np.ndarray:
        """Evaluate the instantaneous wind velocity vector in the NED frame.

        Args:
            t: Elapsed simulation time [s].
            position: Aircraft position vector ``[x, y, z]`` [m] (NED frame).

        Returns:
            Instantaneous wind velocity ``[wx, wy, wz]`` [m/s] (NED frame).
        """
        return self.wind_profile.update(t, position)
