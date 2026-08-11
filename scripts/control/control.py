import math
from typing import Any
import numpy as np


class CascadeController:
    def __init__(self, ac_model: Any) -> None:
        # Current State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)
        self.omega = np.zeros(3)

        # Desired State
        self.des_position = np.zeros(3)
        self.des_velocity = np.zeros(3)
        self.des_omega = np.zeros(3)
        self.des_angles = np.zeros(3)

        # Acceleration
        self.accel = np.zeros(3)

        # Integral Parameters
        self.vel_int_error = np.zeros(3)
        self.rate_int_error = np.zeros(3)
        self.vel_max_integral = 5.0        
        self.rate_max_integral = 0.5   

        # Derivative Parameters
        self.prev_omega = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.filtered_rate_derivative = np.zeros(3)
        self.filtered_vel_derivative = np.zeros(3)

        # ===== Gains: Angular Rate (PID) ===== #
        self.kp_rate = np.array([3.0, 3.0, 2.5])
        self.ki_rate = np.array([0.5, 0.5, 0.5])  
        self.kd_rate = np.array([0.15, 0.15, 0.1])

        # ===== Gains: Attitude (P) ===== #
        self.kp_att = np.array([4.0, 4.0, 2.5])

        # ===== Gains: Velocity (PID) ===== #
        self.kp_vel = np.array([2.0, 2.0, 2.0])
        self.ki_vel = np.array([0.8, 0.8, 0.8])
        self.kd_vel = np.array([0.05, 0.05, 0.05])

        # ===== Gains: Position (P) ===== #
        self.kp_pos = np.array([1.0, 1.0, 1.5])

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
        self.rate_int_error.fill(0.0)
        self.vel_int_error.fill(0.0)
        self.filtered_rate_derivative.fill(0.0)
        self.filtered_vel_derivative.fill(0.0)

    def update_state(self, state: np.ndarray) -> None:
        self.velocity = state[0:3]
        self.omega = state[3:6]
        self.angles = state[6:9]
        self.position = state[9:12]

    def desired_state(self, trajectory: np.ndarray) -> None:
        self.des_position[:] = trajectory[0:3]
        self.des_angles[2] = trajectory[3]

    def position_control(self):
        # Erro
        error = self.des_position - self.position
        vel = self.kp_pos * error

        # Velocidade em Z
        self.des_velocity[2] = max(self.min_z_vel, min(vel[2], self.max_z_vel))

        # Velocidade horizontal XY
        norm_vel_xy = math.hypot(vel[0], vel[1])
        if norm_vel_xy > self.cruise_vel:
            vel[0:2] = (vel[0:2] / norm_vel_xy) * self.cruise_vel

        # Velocidades X e Y decompostas
        psi = self.angles[2]
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        x_vel = (vel[0] * c_psi) + (vel[1] * s_psi) 
        y_vel = -(vel[0] * s_psi) + (vel[1] * c_psi) 
        self.des_velocity[0:2] = x_vel, y_vel

    def velocity_control(self, dt: float):
        error = self.des_velocity[0:3] - self.velocity[0:3]

        # Z error 
        self.vel_int_error[2] += (error[2] * dt)
        self.vel_int_error[2] = max(-self.vel_max_integral, 
                                   min(self.vel_int_error[2], self.vel_max_integral))

        # XY error
        self.vel_int_error[0:2] += (error[0:2] * dt)
        self.vel_int_error[0:2] = np.clip(self.vel_int_error[0:2], 
                                        -self.vel_max_integral, self.vel_max_integral)

        raw_derivative = -(self.velocity - self.prev_vel) / dt

        alpha_vel = 0.2
        self.filtered_vel_derivative = (
                    (alpha_vel * raw_derivative)
                    + ((1.0 - alpha_vel) * self.filtered_vel_derivative))

        self.accel[2] = (
            (self.kp_vel[2] * error[2])
            + (self.ki_vel[2] * self.vel_int_error[2])
            + (self.kd_vel[2] * self.filtered_vel_derivative[2])
        )
        
        self.accel[0:2] = (
            (self.kp_vel[0:2] * error[0:2])
            + (self.ki_vel[0:2] * self.vel_int_error[0:2]) 
            + (self.kd_vel[0:2] * self.filtered_vel_derivative[0:2])
        )

        self.prev_vel[0:3] = self.velocity[0:3]

    def accel_to_attitude(self) -> float:
        g = 9.81

        des_pitch = np.arcsin(-self.accel[0] / g)
        des_roll = np.arcsin(self.accel[1] / g)

        self.des_angles[1] = max(-self.max_tilt, min(des_pitch, self.max_tilt))
        self.des_angles[0] = max(-self.max_tilt, min(des_roll, self.max_tilt))

        current_roll, current_pitch = self.angles[0], self.angles[1]
        c_r, c_p = np.cos(current_roll), np.cos(current_pitch)

        thrust = self.mass * (9.81 - self.accel[2]) / (c_r * c_p)
        thrust = max(10.0, min(thrust, 180.0))

        return thrust

    def angle_control(self) -> None:
        # Error
        error = self.des_angles - self.angles
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))

        # Omega
        omega_calc = self.kp_att * error
        omega_calc = np.clip(omega_calc, -self.max_rate, self.max_rate)
        self.des_omega[:] = omega_calc

    def angular_rate_control(self, dt: float) -> np.ndarray:
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

        torque = (
            (self.kp_rate * error)
            + (self.ki_rate * self.rate_int_error)
            + (self.kd_rate * self.filtered_rate_derivative)
        )

        self.prev_omega[0:3] = self.omega[0:3]
        return torque