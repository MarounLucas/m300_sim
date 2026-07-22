"""Aerodynamic simulation conditions management module.

Acts as a wrapper for the external RotorPy library, allowing the instantiation
of different wind profiles (constant, Dryden gusts, sinusoidal) and obtaining
disturbance vectors at each integration step.

Author: Lucas Maroun de Almeida
"""

from typing import Any
import numpy as np

from rotorpy.wind.default_winds import ConstantWind, NoWind, SinusoidWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP
# from rotorpy.wind.spatial_winds import WindTunnel  # To be added in the future


class WindManager:
    """Manager for stochastic and constant aerodynamic wind models.

    Converts user-provided physical parameters into Inertial Cartesian (NED)
    components and instantiates the appropriate wind profile to inject
    drag forces into the simulation.
    """

    def __init__(
        self,
        wind_type: str = "none",
        magnitude: float = 0.0,
        heading: float = 0.0,
        elevation: float = 0.0,
        gust_magnitude: float = 0.0
    ) -> None:
        """Initializes the wind manager and pre-calculates Cartesian components.

        Args:
            wind_type (str): Wind model ('none', 'constant', 'dryden', 
                'dryden_lp', 'sinusoid').
            magnitude (float): Mean linear wind speed (m/s).
            heading (float): Horizontal wind direction angle (radians).
            elevation (float): Vertical wind elevation angle (radians).
            gust_magnitude (float): Maximum gust magnitude for stochastic models (m/s).
        """
        self.wind_type = wind_type

        s_wh = np.sin(heading)
        c_wh = np.cos(heading)
        s_we = np.sin(elevation)
        c_we = np.cos(elevation)

        # Velocity in the NED coordinate system
        self.wx_ned = magnitude * c_wh * c_we
        self.wy_ned = magnitude * s_wh * c_we
        self.wz_ned = -magnitude * s_we

        # Empirical estimation of standard deviation (sigma) for the Dryden model
        # Based on the difference between the maximum gust intensity and the mean
        sigma_total = (
            max(0.0, (gust_magnitude - magnitude) / 3.0)
            if gust_magnitude > magnitude
            else gust_magnitude
        )

        # Gust standard deviation projected onto the axes
        self.sig_wx = abs(sigma_total * c_wh * c_we)
        self.sig_wy = abs(sigma_total * s_wh * c_we)
        self.sig_wz = abs(sigma_total * s_we)

        self.wind_profile = self._build_profile()

    def _build_profile(self) -> Any:
        """Evaluates the selected wind type and builds the RotorPy object.

        Returns:
            Any: A base wind object containing the `.update()` method.
        """
        if self.wind_type == "constant":
            return ConstantWind(wx=self.wx_ned, wy=self.wy_ned, wz=self.wz_ned)

        if self.wind_type == "dryden":
            avg_wind = np.array([self.wx_ned, self.wy_ned, self.wz_ned])
            sig_wind = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            return DrydenGust(avg_wind=avg_wind, sig_wind=sig_wind, altitude=10.0)

        if self.wind_type == "dryden_lp":
            avg_wind = np.array([self.wx_ned, self.wy_ned, self.wz_ned])
            sig_wind = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            return DrydenGustLP(
                avg_wind=avg_wind, sig_wind=sig_wind, altitude=10.0, tau=0.5
            )

        if self.wind_type == "sinusoid":
            amplitudes = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            frequencies = np.array([0.5, 0.5, 0.5])  # Default oscillatory frequency
            return SinusoidWind(amplitudes=amplitudes, frequencies=frequencies)

        return NoWind()

    def get_wind(self, t: float, position: np.ndarray) -> np.ndarray:
        """Calculates and extracts the instantaneous wind vector.

        Args:
            t (float): Total elapsed global simulation time (seconds).
            position (np.ndarray): Current position vector of the aircraft.

        Returns:
            np.ndarray: Instantaneous wind vector in NED coordinates (m/s).
        """
        return self.wind_profile.update(t, position)