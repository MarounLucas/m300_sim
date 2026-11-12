"""Cascade controller module for the quadcopter.

Heavily inspired by the PX4 control architecture. The system performs control
by dividing it into successive loops: position (P), velocity (PID),
attitude/angle (P), and angular rate (PID).

Author: Lucas Maroun de Almeida
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

import numpy as np


class CascadeController:
    """Cascade controller based on the PX4 multi-loop architecture.

    Controls are divided into successive blocks: position (P), velocity (PID),
    angle (P), and angular rate (PID). Supports operating at a single unified
    frequency (1000 Hz) or at distinct frequencies for each loop.

    Attributes:
        position (np.ndarray): Current 3D position [x, y, z] [m] (NED frame).
        velocity (np.ndarray): Current 3D linear velocity [u, v, w] [m/s] (level frame).
        angles (np.ndarray): Current Euler angles [roll, pitch, yaw] [rad].
        omega (np.ndarray): Current angular rates [p, q, r] [rad/s] (body frame).
    """

    def __init__(self, ac_model: Any) -> None:
        """Initialize the controller parameters, gains, and physical limits."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)
        self.omega = np.zeros(3)

        self.des_position = np.zeros(3)
        self.des_velocity = np.zeros(3)
        self.des_velocity_ff = np.zeros(3)
        self.des_omega = np.zeros(3)
        self.des_angles = np.zeros(3)

        self.a_xy = np.zeros(2)
        self.a_z = 0.0

        self.t_total = 0.0

        self.rate_int_error = np.zeros(3)
        self.xy_vel_int_error = np.zeros(2)
        self.z_vel_int_error = 0.0
        self.max_integral = 2.0
        self.rate_max_integral = 0.5

        self.prev_omega = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.filtered_rate_derivative = np.zeros(3)
        self.filtered_xy_vel_derivative = np.zeros(2)
        self.filtered_z_vel_derivative = 0.0

        self.manual_vel_ref = np.zeros(3)
        self.manual_hold_xy = True
        self.manual_hold_z = True

        self.kp_rate = np.array([3.0, 3.0, 2.5])
        self.ki_rate = np.array([0.5, 0.5, 0.5])
        self.kd_rate = np.array([0.25, 0.25, 0.1])

        self.kp_att = np.array([3.0, 3.0, 2.5])

        self.kp_xy_vel = np.array([2.0, 2.0])
        self.ki_xy_vel = np.array([0.2, 0.2])
        self.kd_xy_vel = np.array([0.25, 0.25])

        self.kp_z_vel = 2.5
        self.ki_z_vel = 0.6
        self.kd_z_vel = 0.3

        self.kp_xy_pos = np.array([1.2, 1.2])
        self.kp_z_pos = 1.5

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
            self.max_yaw_rate,
        ])

    def reset_integrals(self) -> None:
        """Reset integral and derivative memories and sync manual guidance."""
        self.rate_int_error.fill(0.0)
        self.xy_vel_int_error.fill(0.0)
        self.z_vel_int_error = 0.0
        self.filtered_rate_derivative.fill(0.0)
        self.filtered_xy_vel_derivative.fill(0.0)
        self.filtered_z_vel_derivative = 0.0
        self.manual_vel_ref[:] = self.velocity[:]
        self.manual_hold_xy = True
        self.manual_hold_z = True

    def update_state(self, state: np.ndarray) -> None:
        """Convert body-frame velocities into the horizontal level frame across all axes."""
        u, v, w = state[0], state[1], state[2]
        self.omega = state[3:6].copy()
        self.angles = state[6:9].copy()
        self.position = state[9:12].copy()

        phi, theta = self.angles[0], self.angles[1]
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)

        self.velocity[0] = u * c_theta + v * (s_phi * s_theta) + w * (c_phi * s_theta)
        self.velocity[1] = (v * c_phi) - (w * s_phi)
        self.velocity[2] = -u * s_theta + v * (s_phi * c_theta) + w * (c_phi * c_theta)

    def update_auto_guidance(self) -> None:
        """Calculate desired velocity from the reference trajectory in autonomous mode."""
        self._xy_pos_control()
        self._z_pos_control()

    def update_manual_guidance(self, manual_cmd: List[float], dt: float) -> None:
        """Apply slew-rate limiting and position/altitude hold in manual flight mode."""
        x_cmd, y_cmd, z_cmd, yaw_cmd = (
            manual_cmd[0],
            manual_cmd[1],
            manual_cmd[2],
            manual_cmd[3],
        )

        stick_norm = math.hypot(x_cmd, y_cmd)
        if stick_norm > 1.0:
            x_cmd /= stick_norm
            y_cmd /= stick_norm

        max_man_xy_vel = max(self.cruise_vel, 12.0)
        target_vx = x_cmd * max_man_xy_vel
        target_vy = y_cmd * max_man_xy_vel

        if stick_norm > 1e-3:
            self.manual_hold_xy = False
            for i, tv in enumerate([target_vx, target_vy]):
                diff = tv - self.manual_vel_ref[i]
                rate = np.clip(diff / 0.6, -2.5, 2.5)
                self.manual_vel_ref[i] += rate * dt
            self.des_velocity[0:2] = self.manual_vel_ref[0:2]
            self.des_position[0:2] = self.position[0:2]
        else:
            if not self.manual_hold_xy:
                for i in range(2):
                    diff = 0.0 - self.manual_vel_ref[i]
                    rate = np.clip(diff / 0.5, -2.8, 2.8)
                    self.manual_vel_ref[i] += rate * dt
                self.des_velocity[0:2] = self.manual_vel_ref[0:2]
                self.des_position[0:2] = self.position[0:2]

                if (
                    math.hypot(self.manual_vel_ref[0], self.manual_vel_ref[1]) < 0.05
                    and math.hypot(self.velocity[0], self.velocity[1]) < 0.20
                ):
                    self.manual_hold_xy = True
                    self.manual_vel_ref[0:2] = 0.0
                    self.des_position[0:2] = self.position[0:2]

            if self.manual_hold_xy:
                self.des_velocity_ff[0:2] = 0.0
                self._xy_pos_control()

        if abs(z_cmd) > 1e-3:
            self.manual_hold_z = False
            max_z_spd = abs(self.min_z_vel) if z_cmd < 0.0 else self.max_z_vel
            target_vz = z_cmd * max_z_spd
            diff_z = target_vz - self.manual_vel_ref[2]
            rate_z = np.clip(diff_z / 0.4, -3.0, 3.0)
            self.manual_vel_ref[2] += rate_z * dt
            self.des_velocity[2] = self.manual_vel_ref[2]
            self.des_position[2] = self.position[2]
        else:
            if not self.manual_hold_z:
                diff_z = 0.0 - self.manual_vel_ref[2]
                rate_z = np.clip(diff_z / 0.35, -3.5, 3.5)
                self.manual_vel_ref[2] += rate_z * dt
                self.des_velocity[2] = self.manual_vel_ref[2]
                self.des_position[2] = self.position[2]

                if abs(self.manual_vel_ref[2]) < 0.05 and abs(self.velocity[2]) < 0.15:
                    self.manual_hold_z = True
                    self.manual_vel_ref[2] = 0.0
                    self.des_position[2] = self.position[2]

            if self.manual_hold_z:
                self.des_velocity_ff[2] = 0.0
                self._z_pos_control()

        self.des_angles[2] += (yaw_cmd * self.max_yaw_rate) * dt

    def desired_state(self, trajectory: np.ndarray, vel_ff: np.ndarray) -> None:
        """Set the desired state from the current trajectory waypoint and velocity feedforward."""
        self.des_position[:] = trajectory[0:3]
        self.des_angles[2] = trajectory[3]
        self.des_velocity_ff[:] = vel_ff

    def control(self) -> Tuple[float, np.ndarray]:
        """Execute all cascade control loops simultaneously at 1000 Hz."""
        self._xy_pos_control()
        self._z_pos_control()

        self._xy_vel_control(dt=0.001)
        self._z_vel_control(dt=0.001)

        u_total = self._accel_to_attitude()

        self._angle_control()

        tau = self._angular_rate_control(dt=0.001)

        return u_total, tau

    def control_px4(self, tick: int) -> Tuple[float, np.ndarray]:
        """Execute the cascade control loops at distinct multi-rate frequencies."""
        if tick % 20 == 0:
            self._xy_pos_control()
            self._z_pos_control()

            self._xy_vel_control(dt=0.02)
            self._z_vel_control(dt=0.02)

            self.t_total = self._accel_to_attitude()

        if tick % 4 == 0:
            self._angle_control()

        tau = self._angular_rate_control(dt=0.001)

        return self.t_total, tau

    def _xy_pos_control(self) -> None:
        """Calculate desired horizontal velocity and rotate into the horizontal level frame."""
        error_ned = self.des_position[0:2] - self.position[0:2]
        vel_cmd_ned = (self.kp_xy_pos * error_ned) + self.des_velocity_ff[0:2]

        norm_horizontal = math.hypot(vel_cmd_ned[0], vel_cmd_ned[1])
        if norm_horizontal > self.cruise_vel:
            vel_cmd_ned = (vel_cmd_ned / norm_horizontal) * self.cruise_vel

        psi = self.angles[2]
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        self.des_velocity[0] = vel_cmd_ned[0] * c_psi + vel_cmd_ned[1] * s_psi
        self.des_velocity[1] = -vel_cmd_ned[0] * s_psi + vel_cmd_ned[1] * c_psi

    def _z_pos_control(self) -> None:
        """Calculate desired vertical velocity including trajectory feedforward."""
        error = self.des_position[2] - self.position[2]
        z_vel = (self.kp_z_pos * error) + self.des_velocity_ff[2]
        self.des_velocity[2] = max(self.min_z_vel, min(z_vel, self.max_z_vel))

    def _z_vel_control(self, dt: float) -> None:
        """Calculate the required vertical acceleration to reach the target velocity."""
        error = self.des_velocity[2] - self.velocity[2]

        self.z_vel_int_error = self.z_vel_int_error + (error * dt)
        self.z_vel_int_error = max(
            -self.max_integral, min(self.z_vel_int_error, self.max_integral)
        )

        raw_derivative = -(self.velocity[2] - self.prev_vel[2]) / dt

        alpha_vel = 0.3
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
        """Calculate horizontal accelerations with dynamic anti-windup and drag feedforward."""
        error = self.des_velocity[0:2] - self.velocity[0:2]

        effective_g = max(4.0, 9.81 - self.a_z)
        accel_max = effective_g * math.tan(self.max_tilt)
        accel_cmd_norm = math.hypot(self.a_xy[0], self.a_xy[1])

        if accel_cmd_norm < accel_max or (np.dot(error, self.a_xy) < 0):
            self.xy_vel_int_error += error * dt

        self.xy_vel_int_error = np.clip(
            self.xy_vel_int_error, -self.max_integral, self.max_integral
        )

        raw_derivative = -(self.velocity[0:2] - self.prev_vel[0:2]) / dt

        alpha_vel = 0.3
        self.filtered_xy_vel_derivative = (
            (alpha_vel * raw_derivative)
            + ((1.0 - alpha_vel) * self.filtered_xy_vel_derivative)
        )

        # Aerodynamic drag feedforward (cd = 0.25) to reach high cruise speed without integral lag.
        drag_ff = (0.25 / self.mass) * self.des_velocity[0:2]

        self.a_xy = (
            (self.kp_xy_vel * error)
            + (self.ki_xy_vel * self.xy_vel_int_error)
            + (self.kd_xy_vel * self.filtered_xy_vel_derivative)
            + drag_ff
        )

        self.prev_vel[0:2] = self.velocity[0:2]

    def _accel_to_attitude(self) -> float:
        """Convert desired accelerations into attitude angles and tilt-compensated thrust."""
        g = 9.81

        thrust_z = max(0.5 * g, g - self.a_z)
        thrust_req_z = self.mass * thrust_z

        safe_max_tilt = self.max_tilt
        if thrust_req_z > 150.0:
            safe_max_tilt = self.max_tilt * 0.5

        max_a_xy = thrust_z * math.tan(safe_max_tilt)
        a_xy_norm = math.hypot(self.a_xy[0], self.a_xy[1])
        if a_xy_norm > max_a_xy:
            self.a_xy[0] = (self.a_xy[0] / a_xy_norm) * max_a_xy
            self.a_xy[1] = (self.a_xy[1] / a_xy_norm) * max_a_xy

        norm_f = math.sqrt(self.a_xy[0] ** 2 + self.a_xy[1] ** 2 + thrust_z**2)

        self.des_angles[1] = np.arcsin(-self.a_xy[0] / norm_f)
        self.des_angles[0] = np.arcsin(self.a_xy[1] / norm_f)

        # Compensate vertical thrust for the instantaneous physical tilt of the drone.
        current_roll, current_pitch = self.angles[0], self.angles[1]
        tilt_cos = max(0.5, math.cos(current_roll) * math.cos(current_pitch))

        u_total = thrust_req_z / tilt_cos
        return float(np.clip(u_total, 10.0, 180.0))

    def _angle_control(self) -> None:
        """Calculate the angular rate required to reach the desired attitude."""
        error = self.des_angles - self.angles

        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))

        omega_calc = self.kp_att * error
        omega_calc = np.clip(omega_calc, -self.max_rate, self.max_rate)

        self.des_omega[:] = omega_calc

    def _angular_rate_control(self, dt: float) -> np.ndarray:
        """Calculate the motor torques required to track the desired angular rate."""
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

        self.prev_omega = self.omega.copy()

        return tau
