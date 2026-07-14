"""
UAV trajectory planning module. 

This module implements waypoint path interpolation using 5th-order 
polynomial trajectories to ensure smooth transitions with zero velocity 
and acceleration at the segment boundaries.

Author: Lucas Maroun de Almeida
"""

import numpy as np

class PathManager:
    """
    Manages the flight trajectory by dividing waypoints into segments.

    Calculates the required time for each segment based on spatial velocity 
    constraints and pre-computes the polynomial coefficients for the entire mission.

    Attributes
    ----------
    waypoints : list of np.ndarray
        List of target coordinates in 3D space [x, y, z].
    xy_velocity : float
        Cruising horizontal velocity in m/s.
    z_velocity : float
        Cruising vertical velocity in m/s.
    time_safety_factor : float
        Multiplier applied to segment times to ensure trajectory smoothness.
    yaw_mode : str
        The active yaw control strategy ('none', 'forward', or 'target').
    yaw_target : np.ndarray or None
        The [x, y] coordinates the UAV should face when in 'target' mode.
    """
    
    def __init__(self, waypoints: list, yaw_mode: str = 'none', yaw_target: list = None,
                 xy_velocity: float = 9.2, z_velocity: float = 2.0, time_safety_factor: float = 2.6) -> None:
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.xy_velocity = xy_velocity
        self.z_velocity = z_velocity
        self.time_safety_factor = time_safety_factor
        
        self.segments = []
        self.segment_times = []
        self.start_times = []
        self.total_time = 0.0
        self.segment_coefficients = []

        self.yaw_mode = yaw_mode
        self.yaw_target = np.array(yaw_target, dtype=float) if yaw_target else None
        self.last_yaw = 0.0

    def calculate_segment_time(self, q0: np.ndarray, qf: np.ndarray) -> float:
        """
        Estimates the required time to traverse a segment based on velocity limits.

        Evaluates the displacement in the XY plane and Z axis independently, 
        selecting the maximum time required to satisfy both velocity constraints.

        Parameters
        ----------
        q0 : np.ndarray
            The starting coordinate of the segment [x, y, z].
        qf : np.ndarray
            The target coordinate of the segment [x, y, z].

        Returns
        -------
        float
            The estimated travel time in seconds, adjusted by the safety factor.
        """
        dist_xy = np.linalg.norm(qf[0:2] - q0[0:2])
        dist_z = np.abs(qf[2] - q0[2])
        
        time_xy = dist_xy / self.xy_velocity 
        time_z = dist_z / self.z_velocity 
        
        time = max(time_xy, time_z)
        
        if time == 0:
            return 0.1 
        
        if time < 5: 
            time = 5
            
        return time * self.time_safety_factor

    def generate_path(self) -> None:
        """
        Generates the mathematical path for all specified waypoints.

        Iterates through the waypoint list, initializes 5th-order polynomial 
        segments for each leg, and solves for the trajectory coefficients.
        """
        for n in range(len(self.waypoints) - 1):
            q0 = self.waypoints[n]
            qf = self.waypoints[n+1]

            t_seg = self.calculate_segment_time(q0, qf)
            self.start_times.append(self.total_time)
            self.segment_times.append(t_seg)
            self.total_time += t_seg

            traj = PolinomialTraj(q0[0:3], qf[0:3], t_seg)
            self.segments.append(traj)
            
            coefficient = traj.generate_path()
            self.segment_coefficients.append(coefficient)

        print(f"Mission Generated: {len(self.segments)} segments | Total Time: {self.total_time:.2f}s")

    def calculate_yaw(self, idx: int, t_local: float, current_pos: np.ndarray) -> float:
        """
        Calculates the desired yaw angle based on the selected flight mode.

        Parameters
        ----------
        idx : int
            The index of the current trajectory segment.
        t_local : float
            The elapsed time within the current segment.
        current_pos : np.ndarray
            The current spatial position of the UAV [x, y, z].

        Returns
        -------
        float
            The target yaw angle in radians.

        Raises
        ------
        ValueError
            If 'target' mode is selected but no target coordinates are provided.
        """
        if self.yaw_mode == 'forward':
            vel = self.segments[idx].sample_velocity(t_local)
            if np.linalg.norm(vel[0:2]) < 1e-3:
                return self.last_yaw
            
            self.last_yaw = float(np.arctan2(vel[1], vel[0]))
            return self.last_yaw

        elif self.yaw_mode == 'target':
            if self.yaw_target is None:
                raise ValueError("yaw_target must be defined when using 'target' mode.")
            
            dx = self.yaw_target[0] - current_pos[0]
            dy = self.yaw_target[1] - current_pos[1]
            self.last_yaw = float(np.arctan2(dy, dx))
            return self.last_yaw

        return 0.0
    
    def get_desired_state(self, t_global: float) -> tuple:
        """
        Determines the expected UAV state for a given simulation timeframe.

        Parameters
        ----------
        t_global : float
            The total elapsed time since the mission started, in seconds.

        Returns
        -------
        tuple
            A tuple containing:
            - desire_position (np.ndarray): The target spatial coordinates [x, y, z].
            - desire_velocity (np.ndarray): The target velocity vector [u, v, w].
            - desire_yaw (float): The target yaw angle in radians.
            - idx (int): The index of the active segment.
        """
        if t_global >= self.total_time:
            final_pos = self.waypoints[-1][0:3]
            # Returns a zero velocity array for the final state to maintain tuple structure consistency
            return final_pos, np.zeros(3), self.last_yaw, len(self.segments) - 1
        
        idx = 0
        for i, start_t in enumerate(self.start_times):
            if t_global >= start_t:
                idx = i
            else:
                break

        t_local = t_global - self.start_times[idx]
        desire_position = self.segments[idx].sample_position(t_local)
        desire_velocity = self.segments[idx].sample_velocity(t_local)
        desire_yaw = self.calculate_yaw(idx, t_local, desire_position)

        return desire_position, desire_velocity, desire_yaw, idx


class PolinomialTraj:
    """
    Constructs and solves a 5th-order polynomial trajectory for a 3D segment.

    Boundary conditions enforce that the start and end positions are reached 
    with zero velocity and zero acceleration, ensuring smooth kinematics.
    """

    def __init__(self, q0: np.ndarray, qf: np.ndarray, t: float) -> None:
        """
        Initializes the polynomial segment with zero initial/final derivatives.

        Parameters
        ----------
        q0 : np.ndarray
            The starting position vector [x, y, z].
        qf : np.ndarray
            The final position vector [x, y, z].
        t : float
            The duration allocated to complete the segment, in seconds.
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
        self.des_position = np.zeros(3)
        self.des_velocity = np.zeros(3)
        self.x = np.zeros((6, 3))

    def generate_path(self) -> np.ndarray:
        """
        Solves the linear system to find the polynomial coefficients.

        Formulates the 6 boundary conditions (position, velocity, acceleration 
        at both extremes) into an Ax = b system and solves for x.

        Returns
        -------
        np.ndarray
            A (6x3) matrix containing the polynomial coefficients for X, Y, and Z.

        Notes
        -----
        Future optimization: Implement closed-form algebraic equations to 
        bypass the computational cost of matrix inversion.
        """
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, self.time, self.time**2, self.time**3, self.time**4, self.time**5],
            [0, 1, 2*self.time, 3*self.time**2, 4*self.time**3, 5*self.time**4],
            [0, 0, 2, 6*self.time, 12*self.time**2, 20*self.time**3]
        ])
        b = np.array([self.q0, self.dq0, self.ddq0, self.qf, self.dqf, self.ddqf])
        
        self.x = np.linalg.solve(A, b)
        return self.x
    
    def sample_position(self, t_atual: float) -> np.ndarray:
        """
        Evaluates the 5th-order polynomial at a specific local time.

        Parameters
        ----------
        t_atual : float
            Elapsed time since the beginning of the segment, in seconds.

        Returns
        -------
        np.ndarray
            The calculated instantaneous position [x, y, z].
        """
        self.powers[0] = 1.0
        self.powers[1] = t_atual
        self.powers[2] = t_atual**2
        self.powers[3] = t_atual**3
        self.powers[4] = t_atual**4
        self.powers[5] = t_atual**5
        
        np.dot(self.powers, self.x, out=self.des_position)
        return self.des_position

    def sample_velocity(self, t_atual: float) -> np.ndarray:
        """
        Evaluates the first derivative of the polynomial at a specific local time.

        Parameters
        ----------
        t_atual : float
            Elapsed time since the beginning of the segment, in seconds.

        Returns
        -------
        np.ndarray
            The calculated instantaneous velocity [u, v, w].
        """
        self.vel_powers[0] = 0.0
        self.vel_powers[1] = 1.0
        self.vel_powers[2] = 2.0 * t_atual
        self.vel_powers[3] = 3.0 * (t_atual**2)
        self.vel_powers[4] = 4.0 * (t_atual**3)
        self.vel_powers[5] = 5.0 * (t_atual**4)
        
        np.dot(self.vel_powers, self.x, out=self.des_velocity)
        return self.des_velocity