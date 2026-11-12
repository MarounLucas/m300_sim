"""UAV trajectory planning with 5th-order polynomial segments."""

from __future__ import annotations

import numpy as np


class PolynomialTrajectory:
    """Construct a 5th-order polynomial trajectory for a 3D segment.

    Boundary conditions enforce that the start and end positions are reached
    with zero velocity and zero acceleration for smooth segment transitions.

    Attributes:
        q0 (np.ndarray): Starting position vector [x, y, z] [m] (NED frame).
        qf (np.ndarray): Final position vector [x, y, z] [m] (NED frame).
        time (float): Segment duration [s].
        x (np.ndarray): Polynomial coefficients matrix of shape (6, 3).
    """

    def __init__(self, q0: np.ndarray, qf: np.ndarray, t: float) -> None:
        """Initialize the segment with zero boundary velocities and accelerations.

        Args:
            q0: Starting position vector [x, y, z] [m] (NED frame).
            qf: Final position vector [x, y, z] [m] (NED frame).
            t: Segment duration [s].
        """
        self.q0 = q0.copy()
        self.dq0 = np.zeros(3)
        self.ddq0 = np.zeros(3)
        self.qf = qf.copy()
        self.dqf = np.zeros(3)
        self.ddqf = np.zeros(3)
        self.time = t

        self.powers = np.zeros(6)
        self.vel_powers = np.zeros(6)
        self.desired_position = np.zeros(3)
        self.desired_velocity = np.zeros(3)
        self.x = np.zeros((6, 3))

    def generate_path(self) -> np.ndarray:
        """Solve the boundary value linear system for the polynomial coefficients.

        Returns:
            Coefficient matrix of shape (6, 3) for the X, Y, and Z axes.
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
        """Evaluate the 5th-order position polynomial at ``current_t``.

        Args:
            current_t: Elapsed time since the segment start [s].

        Returns:
            Instantaneous position [x, y, z] [m] (NED frame).
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
        """Evaluate the first derivative of the polynomial at ``current_t``.

        Args:
            current_t: Elapsed time since the segment start [s].

        Returns:
            Instantaneous velocity [vx, vy, vz] [m/s] (NED frame).
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
    """Manage multi-waypoint trajectories and heading (yaw) references.

    Divides the waypoint list into 5th-order polynomial segments, sizes each
    segment duration from horizontal and vertical speed limits, and computes
    the yaw reference according to the selected heading mode.

    Attributes:
        waypoints (list[np.ndarray]): Waypoint coordinates [x, y, z] [m] (NED frame).
        xy_velocity (float): Nominal horizontal cruising speed [m/s].
        z_velocity (float): Nominal vertical speed [m/s].
        time_safety_factor (float): Duration multiplier to keep peak speeds within limits.
        yaw_mode (str): Active yaw control mode (``'none'``, ``'free'``, ``'forward'``,
            or ``'target'``).
        yaw_target (np.ndarray | None): Horizontal target point [x, y] [m] (NED frame)
            to face when ``yaw_mode == 'target'``.
    """

    def __init__(
        self,
        waypoints: list[list[float]],
        yaw_mode: str = 'none',
        yaw_target: list[float] | np.ndarray | None = None,
        xy_velocity: float = 9.2,
        z_velocity: float = 2.0,
        time_safety_factor: float = 1.875,
    ) -> None:
        """Initialize the trajectory manager with velocity constraints and yaw settings."""
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.xy_velocity = xy_velocity
        self.z_velocity = z_velocity
        self.time_safety_factor = time_safety_factor

        self.segments: list[PolynomialTrajectory] = []
        self.segment_times: list[float] = []
        self.start_times: list[float] = []
        self.total_time = 0.0
        self.segment_coefficients: list[np.ndarray] = []

        self.yaw_mode = yaw_mode
        self.yaw_target: np.ndarray | None = (
            np.asarray(yaw_target[:2], dtype=float).copy()
            if yaw_target is not None and len(yaw_target) >= 2
            else None
        )
        self.last_yaw = 0.0

    def calculate_segment_time(
        self, q0: np.ndarray, qf: np.ndarray
    ) -> float:
        """Estimate the duration required to traverse a segment within speed limits.

        Args:
            q0: Segment start coordinate [x, y, z] [m] (NED frame).
            qf: Segment end coordinate [x, y, z] [m] (NED frame).

        Returns:
            Segment duration [s] scaled by ``time_safety_factor``.
        """
        dist_xy = np.linalg.norm(qf[0:2] - q0[0:2])
        dist_z = np.abs(qf[2] - q0[2])

        time_xy = float(dist_xy / self.xy_velocity)
        time_z = float(dist_z / self.z_velocity)

        time = max(time_xy, time_z)

        # Enforce minimum duration bounds to prevent ill-conditioned Vandermonde matrices
        if time == 0:
            return 0.1
        if time < 5:
            time = 5.0

        return time * self.time_safety_factor

    def generate_path(self) -> None:
        """Build 5th-order polynomial segments across all consecutive waypoints."""
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
            f'Mission Generated: {segment_count} segments | '
            f'Total Time: {self.total_time:.2f}s'
        )

    def calculate_yaw(
        self, idx: int, local_t: float, current_pos: np.ndarray
    ) -> float:
        """Calculate the desired yaw angle in the horizontal XY plane.

        Args:
            idx: Index of the active trajectory segment.
            local_t: Elapsed time within the active segment [s].
            current_pos: Current spatial position [x, y, z] [m] (NED frame).

        Returns:
            Desired yaw angle [rad] (NED frame).

        Raises:
            ValueError: If ``'target'`` mode is active without ``yaw_target``.
        """
        if self.yaw_mode == 'forward':
            vel = self.segments[idx].sample_velocity(local_t)
            # Hold previous heading when horizontal velocity vanishes at segment ends
            if np.linalg.norm(vel[0:2]) < 1e-3:
                return self.last_yaw

            self.last_yaw = float(np.arctan2(vel[1], vel[0]))
            return self.last_yaw

        if self.yaw_mode == 'target':
            if self.yaw_target is None:
                raise ValueError(
                    "yaw_target must be defined when using 'target' mode."
                )

            dx = self.yaw_target[0] - current_pos[0]
            dy = self.yaw_target[1] - current_pos[1]
            # Hold previous heading if the UAV is directly aligned with the target XY point
            if np.hypot(dx, dy) < 1e-3:
                return self.last_yaw

            self.last_yaw = float(np.arctan2(dy, dx))
            return self.last_yaw

        return 0.0

    def get_desired_state(
        self, global_t: float
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Evaluate the desired position, velocity, yaw, and segment index at ``global_t``.

        Args:
            global_t: Total elapsed mission time [s].

        Returns:
            Tuple ``(desired_position, desired_velocity, desired_yaw, idx)`` where
            ``desired_position`` is [x, y, z] [m] (NED frame), ``desired_velocity``
            is [vx, vy, vz] [m/s] (NED frame), ``desired_yaw`` is [rad], and ``idx``
            is the active segment index.
        """
        if global_t >= self.total_time:
            final_pos = self.waypoints[-1][0:3].copy()
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


if __name__ == '__main__':
    sample_waypoints = [
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 10.0],
        [20.0, 0.0, 5.0],
    ]

    path_manager = PathManager(
        waypoints=sample_waypoints,
        yaw_mode='forward',
        xy_velocity=10.0,
        z_velocity=3.0,
    )
    path_manager.generate_path()

    mid_time = path_manager.total_time / 2.0
    pos, vel, yaw, segment_idx = path_manager.get_desired_state(mid_time)

    print(f'\nMid-mission state at t={mid_time:.2f}s:')
    print(f'  Segment Index: {segment_idx}')
    print(f'  Position: {pos}')
    print(f'  Velocity: {vel}')
    print(f'  Yaw: {yaw:.2f} rad')
