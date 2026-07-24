"""UAV trajectory planning module.

This module implements waypoint path interpolation using 5th-order
polynomial trajectories to ensure smooth transitions with zero velocity
and acceleration at the segment boundaries.

Author: Lucas Maroun de Almeida
"""

import numpy as np
from typing import List, Optional, Tuple


class PolynomialTrajectory:
    """Constructs a 5th-order polynomial trajectory for a 3D segment.

    Boundary conditions enforce that the start and end positions are reached
    with zero velocity and zero acceleration, ensuring smooth kinematics.

    Example:
        traj = PolynomialTrajectory(
            q0=np.array([0.0, 0.0, 0.0]),
            qf=np.array([10.0, 10.0, 5.0]),
            t=5.0
        )
        coeffs = traj.generate_path()
        current_pos = traj.sample_position(2.5)

    Attributes:
        q0 (np.ndarray): Starting position vector [x, y, z].
        qf (np.ndarray): Final position vector [x, y, z].
        time (float): Segment duration in seconds.
        x (np.ndarray): Polynomial coefficients matrix (6x3).
    """

    def __init__(self, q0: np.ndarray, qf: np.ndarray, t: float) -> None:
        """Initializes the segment with zero initial and final derivatives.

        Args:
            q0 (np.ndarray): The starting position vector [x, y, z].
            qf (np.ndarray): The final position vector [x, y, z].
            t (float): The duration allocated to complete the segment.
        """
        self.q0 = q0
        self.dq0 = np.zeros(3)
        self.ddq0 = np.zeros(3)
        self.qf = qf
        self.dqf = np.zeros(3)
        self.ddqf = np.zeros(3)
        self.time = t

        self.powers = np.zeros(6)
        self.vel_powers = np.zeros(6)
        self.desired_position = np.zeros(3)
        self.desired_velocity = np.zeros(3)
        self.x = np.zeros((6, 3))

    def generate_path(self) -> np.ndarray:
        """Solves the linear system to find the polynomial coefficients.

        Formulates the 6 boundary conditions (position, velocity,
        acceleration at both extremes) into an Ax = b system and solves it.

        Returns:
            np.ndarray: A (6x3) matrix of polynomial coefficients (X, Y, Z).
        """
        t = self.time
        coeff_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2 * t, 3 * (t**2), 4 * (t**3), 5 * (t**4)],
            [0, 0, 2, 6 * t, 12 * (t**2), 20 * (t**3)],
        ])
        boundaries = np.array([
            self.q0,
            self.dq0,
            self.ddq0,
            self.qf,
            self.dqf,
            self.ddqf,
        ])

        self.x = np.linalg.solve(coeff_matrix, boundaries)
        return self.x

    def sample_position(self, current_t: float) -> np.ndarray:
        """Evaluates the 5th-order polynomial at a specific local time.

        Args:
            current_t (float): Elapsed time since the segment start (seconds).

        Returns:
            np.ndarray: The calculated instantaneous position [x, y, z].
        """
        self.powers[0] = 1.0
        self.powers[1] = current_t
        self.powers[2] = current_t**2
        self.powers[3] = current_t**3
        self.powers[4] = current_t**4
        self.powers[5] = current_t**5

        np.dot(self.powers, self.x, out=self.desired_position)
        return self.desired_position

    def sample_velocity(self, current_t: float) -> np.ndarray:
        """Evaluates the first derivative of the polynomial at a given time.

        Args:
            current_t (float): Elapsed time since the segment start (seconds).

        Returns:
            np.ndarray: The calculated instantaneous velocity [u, v, w].
        """
        self.vel_powers[0] = 0.0
        self.vel_powers[1] = 1.0
        self.vel_powers[2] = 2.0 * current_t
        self.vel_powers[3] = 3.0 * (current_t**2)
        self.vel_powers[4] = 4.0 * (current_t**3)
        self.vel_powers[5] = 5.0 * (current_t**4)

        np.dot(self.vel_powers, self.x, out=self.desired_velocity)
        return self.desired_velocity


class PathManager:
    """Manages the flight trajectory by dividing waypoints into segments.

    Calculates the required time for each segment based on spatial velocity
    constraints and pre-computes the polynomial coefficients.

    Example:
        waypoints = [[0.0, 0.0, 0.0], [10.0, 10.0, 5.0]]
        manager = PathManager(waypoints, yaw_mode="forward")
        manager.generate_path()
        pos, vel, yaw, idx = manager.get_desired_state(2.5)

    Attributes:
        waypoints (List[np.ndarray]): Target coordinates in 3D space [x, y, z].
        xy_velocity (float): Cruising horizontal velocity in m/s.
        z_velocity (float): Cruising vertical velocity in m/s.
        time_safety_factor (float): Multiplier for trajectory smoothness.
        yaw_mode (str): Active yaw control ('none', 'forward', 'target').
        yaw_target (Optional[np.ndarray]): Coordinates to face in target mode.
    """

    def __init__(
        self,
        waypoints: List[List[float]],
        yaw_mode: str = "none",
        yaw_target: Optional[List[float]] = None,
        xy_velocity: float = 9.2,
        z_velocity: float = 2.0,
        time_safety_factor: float = 2.6,
    ) -> None:
        """Initializes the trajectory manager with velocity constraints."""
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.xy_velocity = xy_velocity
        self.z_velocity = z_velocity
        self.time_safety_factor = time_safety_factor

        self.segments: List[PolynomialTrajectory] = []
        self.segment_times: List[float] = []
        self.start_times: List[float] = []
        self.total_time = 0.0
        self.segment_coefficients: List[np.ndarray] = []

        self.yaw_mode = yaw_mode
        self.yaw_target = (
            np.array(yaw_target, dtype=float) if yaw_target else None
        )
        self.last_yaw = 0.0

    def calculate_segment_time(
        self, q0: np.ndarray, qf: np.ndarray
    ) -> float:
        """Estimates required time to traverse a segment based on limits.

        Evaluates the displacement in the XY plane and Z axis independently,
        selecting the maximum time required to satisfy both constraints.

        Args:
            q0 (np.ndarray): The starting coordinate [x, y, z].
            qf (np.ndarray): The target coordinate [x, y, z].

        Returns:
            float: Estimated travel time adjusted by the safety factor.
        """
        dist_xy = np.linalg.norm(qf[0:2] - q0[0:2])
        dist_z = np.abs(qf[2] - q0[2])

        time_xy = float(dist_xy / self.xy_velocity)
        time_z = float(dist_z / self.z_velocity)

        time = max(time_xy, time_z)

        # Enforce minimum time bounds to prevent singularities or jerky motion
        if time == 0:
            return 0.1
        if time < 5:
            time = 5.0

        return time * self.time_safety_factor

    def generate_path(self) -> None:
        """Generates the mathematical path for all specified waypoints.

        Iterates through the waypoint list, initializes 5th-order polynomial
        segments for each leg, and solves for the trajectory coefficients.
        """
        for n in range(len(self.waypoints) - 1):
            q0 = self.waypoints[n]
            qf = self.waypoints[n + 1]

            t_seg = self.calculate_segment_time(q0, qf)
            self.start_times.append(self.total_time)
            self.segment_times.append(t_seg)
            self.total_time += t_seg

            traj = PolynomialTrajectory(q0[0:3], qf[0:3], t_seg)
            self.segments.append(traj)

            coefficient = traj.generate_path()
            self.segment_coefficients.append(coefficient)

        segment_count = len(self.segments)
        print(
            f"Mission Generated: {segment_count} segments | "
            f"Total Time: {self.total_time:.2f}s"
        )

    def calculate_yaw(
        self, idx: int, local_t: float, current_pos: np.ndarray
    ) -> float:
        """Calculates the desired yaw angle based on the flight mode.

        Args:
            idx (int): Index of the current trajectory segment.
            local_t (float): Elapsed time within the current segment.
            current_pos (np.ndarray): Current spatial position [x, y, z].

        Returns:
            float: The target yaw angle in radians.

        Raises:
            ValueError: If 'target' mode lacks target coordinates.
        """
        if self.yaw_mode == "forward":
            vel = self.segments[idx].sample_velocity(local_t)
            # Maintain previous yaw if horizontal velocity is negligible
            if np.linalg.norm(vel[0:2]) < 1e-3:
                return self.last_yaw

            self.last_yaw = float(np.arctan2(vel[1], vel[0]))
            return self.last_yaw

        elif self.yaw_mode == "target":
            if self.yaw_target is None:
                raise ValueError(
                    "yaw_target must be defined when using 'target' mode."
                )

            dx = self.yaw_target[0] - current_pos[0]
            dy = self.yaw_target[1] - current_pos[1]
            self.last_yaw = float(np.arctan2(dy, dx))
            return self.last_yaw

        return 0.0

    def get_desired_state(
        self, global_t: float
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Determines the expected UAV state for a given timeframe.

        Args:
            global_t (float): Total elapsed time since mission start.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, int]: A tuple containing:
                - desired_position (np.ndarray): Target coordinates [x, y, z].
                - desired_velocity (np.ndarray): Target velocity [u, v, w].
                - desired_yaw (float): Target yaw angle in radians.
                - idx (int): Index of the active segment.
        """
        if global_t >= self.total_time:
            final_pos = self.waypoints[-1][0:3]
            return final_pos, np.zeros(3), self.last_yaw, len(self.segments) - 1

        idx = 0
        for i, start_t in enumerate(self.start_times):
            if global_t >= start_t:
                idx = i
            else:
                break

        local_t = global_t - self.start_times[idx]
        desired_position = self.segments[idx].sample_position(local_t)
        desired_velocity = self.segments[idx].sample_velocity(local_t)
        desired_yaw = self.calculate_yaw(idx, local_t, desired_position)

        return desired_position, desired_velocity, desired_yaw, idx


if __name__ == "__main__":
    # Example usage for testing and demonstration
    sample_waypoints = [
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 10.0],
        [20.0, 0.0, 5.0]
    ]
    
    path_manager = PathManager(
        waypoints=sample_waypoints,
        yaw_mode="forward",
        xy_velocity=10.0,
        z_velocity=3.0,
    )
    path_manager.generate_path()

    mid_time = path_manager.total_time / 2.0
    pos, vel, yaw, segment_idx = path_manager.get_desired_state(mid_time)

    print(f"\nMid-mission state at t={mid_time:.2f}s:")
    print(f"  Segment Index: {segment_idx}")
    print(f"  Position: {pos}")
    print(f"  Velocity: {vel}")
    print(f"  Yaw: {yaw:.2f} rad")