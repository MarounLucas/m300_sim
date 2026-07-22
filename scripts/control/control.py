"""Cascade controller module for the quadcopter.

Heavily inspired by the PX4 control architecture. The system performs control 
by dividing it into successive loops: position (P), velocity (PID), 
attitude/angle (P), and angular rate (PID).

Author: Lucas Maroun de Almeida
"""

import math
from typing import Any, Tuple

import numpy as np


class CascadeController:
    """Cascade Controller based on the PX4 formulation.

    Controls are divided into successive blocks: position (P), velocity (PID), 
    angle (P), and angular rate (PID). The main difference from the original 
    PX4 is the ability to operate at a single frequency (1000Hz) or at 
    distinct frequencies for each loop.

    Attributes:
        position (np.ndarray): Current 3D position [x, y, z].
        velocity (np.ndarray): Current 3D linear velocity [u, v, w].
        angles (np.ndarray): Current Euler angles [roll, pitch, yaw].
        omega (np.ndarray): Current angular rates [p, q, r].
    """

    def __init__(self, ac_model: Any) -> None:
        """Initializes the controller parameters, gains, and physical limits.

        Args:
            ac_model (Any): Object containing the aircraft's physical parameters 
                and limits.
        """
        # Current State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)
        self.omega = np.zeros(3)

        # Desired State
        self.des_position = np.zeros(3)
        self.des_velocity = np.zeros(3)
        self.des_velocity_ff = np.zeros(3)
        self.des_omega = np.zeros(3)
        self.des_angles = np.zeros(3)

        self.a_xy = np.zeros(2)
        self.a_z = 0.0

        self.t_total = 0.0

        # Integral Parameters
        self.rate_int_error = np.zeros(3)
        self.xy_vel_int_error = np.zeros(2)
        self.z_vel_int_error = 0.0
        self.max_integral = 5.0        # Prevents wind-up for velocity
        self.rate_max_integral = 0.5   # Prevents wind-up for angular rate

        # Derivative Parameters
        self.prev_omega = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.filtered_rate_derivative = np.zeros(3)
        self.filtered_xy_vel_derivative = np.zeros(2)
        self.filtered_z_vel_derivative = 0.0

        # ===== Gains: Angular Rate (PID) ===== #
        self.kp_rate = np.array([3.0, 3.0, 2.5])
        self.ki_rate = np.array([0.5, 0.5, 0.5])  # Kept low to avoid overload
        self.kd_rate = np.array([0.15, 0.15, 0.1])

        # ===== Gains: Attitude (P) ===== #
        self.kp_att = np.array([4.0, 4.0, 2.5])

        # ===== Gains: XY Velocity (PID) ===== #
        self.kp_xy_vel = np.array([2.0, 2.0])
        self.ki_xy_vel = np.array([0.8, 0.8])
        self.kd_xy_vel = np.array([0.05, 0.05])

        # ===== Gains: Z Velocity (PID) ===== #
        self.kp_z_vel = 2.0
        self.ki_z_vel = 0.8
        self.kd_z_vel = 0.4

        # ===== Gains: XY and Z Position (P) ===== #
        # Since Feed-Forward is active, the position loop can be more reactive
        self.kp_xy_pos = np.array([1.0, 1.0])
        self.kp_z_pos = 1.5

        # ===== Aircraft Parameters ===== #
        self.mass = ac_model.mass
        self.min_z_vel = ac_model.max_ascent_speed
        self.max_z_vel = ac_model.max_descent_speed
        self.cruise_vel = ac_model.cruise_speed
        self.max_tilt = ac_model.max_tilt_angle
        self.max_roll_pitch_rate = ac_model.max_roll_pitch_rate
        self.max_yaw_rate = ac_model.max_yaw_rate
        self.max_rate = np.array([
            self.max_roll_pitch_rate,
            self.max_roll_pitch_rate,
            self.max_yaw_rate
        ])

    def reset_integrals(self) -> None:
        """Resets integral and derivative memories to avoid bumps on mode switch."""
        self.rate_int_error.fill(0.0)
        self.xy_vel_int_error.fill(0.0)
        self.z_vel_int_error = 0.0
        self.filtered_rate_derivative.fill(0.0)
        self.filtered_xy_vel_derivative.fill(0.0)
        self.filtered_z_vel_derivative = 0.0

    def update_state(self, state: np.ndarray) -> None:
        """Receives and updates the current state vector of the aircraft.

        Args:
            state (np.ndarray): Current state vector containing linear velocities (0:3), 
                angular rates (3:6), Euler angles (6:9), and XYZ positions (9:12).
        """
        self.velocity = state[0:3]
        self.omega = state[3:6]
        self.angles = state[6:9]
        self.position = state[9:12]

    def desired_state(self, trajectory: np.ndarray, vel_ff: np.ndarray) -> None:
        """Sets the desired state based on the current trajectory waypoint.

        Args:
            trajectory (np.ndarray): Vector containing the current waypoint. 
                Contains desired positions (x, y, z) and yaw angle.
            vel_ff (np.ndarray): Feedforward velocity vector.
        """
        self.des_position[:] = trajectory[0:3]
        self.des_angles[2] = trajectory[3]
        self.des_velocity_ff.fill(0.0)

    def control(self) -> Tuple[float, np.ndarray]:
        """Executes all cascade control loops simultaneously.

        Sequentially calls the position, velocity, attitude, and angular rate 
        controllers assuming a fixed high-frequency loop (e.g., 1000Hz).

        Returns:
            Tuple[float, np.ndarray]: Total requested thrust and a 3D vector 
                of requested torques (Roll, Pitch, Yaw).
        """
        self._xy_pos_control()
        self._z_pos_control()

        # Assuming 1000Hz execution for the unified loop
        self._xy_vel_control(dt=0.001)
        self._z_vel_control(dt=0.001)

        u_total = self._accel_to_attitude()

        self._angle_control()

        tau = self._angular_rate_control(dt=0.001)

        return u_total, tau

    def control_px4(self, tick: int) -> Tuple[float, np.ndarray]:
        """Executes the cascade control loops at distinct frequencies.

        Args:
            tick (int): The current simulation tick, used to trigger slower loops.

        Returns:
            Tuple[float, np.ndarray]: Total requested thrust and a 3D vector 
                of requested torques (Roll, Pitch, Yaw).
        """
        if tick % 20 == 0:  # 50Hz Loop
            self._xy_pos_control()
            self._z_pos_control()

            self._xy_vel_control(dt=0.02)
            self._z_vel_control(dt=0.02)

            self.t_total = self._accel_to_attitude()

        if tick % 4 == 0:  # 250Hz Loop
            self._angle_control()

        # 1000Hz Loop
        tau = self._angular_rate_control(dt=0.001)

        return self.t_total, tau

    def _xy_pos_control(self) -> None:
        """Calculates the required horizontal velocity to reach the target position.

        Control Type: Proportional (P)
        """
        error = self.des_position[0:2] - self.position[0:2]
        xy_vel = self.kp_xy_pos * error

        norm_horizontal = math.hypot(xy_vel[0], xy_vel[1])
        if norm_horizontal > self.cruise_vel:
            xy_vel = (xy_vel / norm_horizontal) * self.cruise_vel

        psi = self.angles[2]
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        u_cmd = xy_vel[0] * c_psi + xy_vel[1] * s_psi
        v_cmd = -xy_vel[0] * s_psi + xy_vel[1] * c_psi

        # Feedforward application (currently zeroed out in desired_state)
        u_ff = self.des_velocity_ff[0] * c_psi + self.des_velocity_ff[1] * s_psi
        v_ff = -self.des_velocity_ff[0] * s_psi + self.des_velocity_ff[1] * c_psi

        self.des_velocity[0] = u_cmd
        self.des_velocity[1] = v_cmd

    def _z_pos_control(self) -> None:
        """Calculates the required vertical velocity to reach the target altitude.

        Control Type: Proportional (P)
        """
        error = self.des_position[2] - self.position[2]
        z_vel = self.kp_z_pos * error
        self.des_velocity[2] = max(self.min_z_vel, min(z_vel, self.max_z_vel))

    def _z_vel_control(self, dt: float) -> None:
        """Calculates the required vertical acceleration to reach the target velocity.

        Control Type: Proportional, Integral, Derivative (PID)

        Args:
            dt (float): The time step of the control loop in seconds.
        """
        error = self.des_velocity[2] - self.velocity[2]

        self.z_vel_int_error = self.z_vel_int_error + (error * dt)
        self.z_vel_int_error = max(
            -self.max_integral, min(self.z_vel_int_error, self.max_integral)
        )

        raw_derivative = -(self.velocity[2] - self.prev_vel[2]) / dt

        alpha_vel = 0.2
        self.filtered_z_vel_derivative = (
            (alpha_vel * raw_derivative)
            + ((1.0 - alpha_vel) * self.filtered_z_vel_derivative)
        )

        self.a_z = (
            (self.kp_z_vel * error)
            + (self.ki_z_vel * self.z_vel_int_error)
            + (self.kd_z_vel * self.filtered_z_vel_derivative)
        )

        self.prev_vel[2] = self.velocity[2]

    def _xy_vel_control(self, dt: float) -> None:
        """Calculates the required horizontal accelerations to reach target velocities.

        Control Type: Proportional, Integral, Derivative (PID)

        Args:
            dt (float): The time step of the control loop in seconds.
        """
        error = self.des_velocity[0:2] - self.velocity[0:2]

        self.xy_vel_int_error = self.xy_vel_int_error + (error * dt)
        self.xy_vel_int_error = np.clip(
            self.xy_vel_int_error, -self.max_integral, self.max_integral
        )

        raw_derivative = -(self.velocity[0:2] - self.prev_vel[0:2]) / dt

        alpha_vel = 0.02
        self.filtered_xy_vel_derivative = (
            (alpha_vel * raw_derivative)
            + ((1.0 - alpha_vel) * self.filtered_xy_vel_derivative)
        )

        self.a_xy = (
            (self.kp_xy_vel * error)
            + (self.ki_xy_vel * self.xy_vel_int_error)
            + (self.kd_xy_vel * self.filtered_xy_vel_derivative)
        )

        self.prev_vel[0:2] = self.velocity[0:2]

    def _accel_to_attitude(self) -> float:
        """Converts desired accelerations into attitude references (roll, pitch).

        Also calculates the total thrust required to balance forces.

        Returns:
            float: The calculated total thrust value.
        """
        g = 9.81

        des_pitch = np.arcsin(-self.a_xy[0] / g)
        des_roll = np.arcsin(self.a_xy[1] / g)

        self.des_angles[1] = max(-self.max_tilt, min(des_pitch, self.max_tilt))
        self.des_angles[0] = max(-self.max_tilt, min(des_roll, self.max_tilt))

        current_roll, current_pitch = self.angles[0], self.angles[1]
        c_r, c_p = np.cos(current_roll), np.cos(current_pitch)

        u_total = self.mass * (9.81 - self.a_z) / (c_r * c_p)
        u_total = max(10.0, min(u_total, 180.0))

        return u_total

    def _angle_control(self) -> None:
        """Calculates the angular rate required to reach the desired attitude.

        Control Type: Proportional (P)
        """
        error = self.des_angles - self.angles

        # Normalize yaw error to [-pi, pi]
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))

        omega_calc = self.kp_att * error
        omega_calc = np.clip(omega_calc, -self.max_rate, self.max_rate)

        self.des_omega[:] = omega_calc

    def _angular_rate_control(self, dt: float) -> np.ndarray:
        """Calculates the torque required by the motors to reach the desired angular rate.

        Control Type: Proportional, Integral, Derivative (PID)

        Args:
            dt (float): The time step of the control loop in seconds.

        Returns:
            np.ndarray: Vector containing the requested torques (tau).
        """
        error = self.des_omega - self.omega

        self.rate_int_error += error * dt
        self.rate_int_error = np.clip(
            self.rate_int_error, -self.rate_max_integral, self.rate_max_integral
        )

        raw_derivative = -(self.omega - self.prev_omega) / dt
        alpha_rate = 0.2
        self.filtered_rate_derivative = (
            (alpha_rate * raw_derivative)
            + ((1.0 - alpha_rate) * self.filtered_rate_derivative)
        )

        tau = (
            (self.kp_rate * error)
            + (self.ki_rate * self.rate_int_error)
            + (self.kd_rate * self.filtered_rate_derivative)
        )

        self.prev_omega = self.omega

        return tau