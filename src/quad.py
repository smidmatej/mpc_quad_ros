 # 
 # This file is part of the mpc_quad_ros distribution (https://github.com/smidmatej/mpc_quad_ros).
 # Copyright (c) 2023 Smid Matej.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

import numpy as np
from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse



class Quadrotor3D:

	def __init__(self, payload=False, drag=False):
		"""
		Initialization of the 3D quadrotor class
		:param noisy: Whether noise is used in the simulation
		:type noisy: bool
		:param drag: Whether to simulate drag or not.
		:type drag: bool
		:param payload: Whether to simulate a payload force in the simulation
		:type payload: bool
		:param motor_noise: Whether non-gaussian noise is considered in the motor inputs
		:type motor_noise: bool
		"""


		# Maximum thrust in Newtons of a thruster when rotating at maximum speed.
		self.max_thrust = 20

		self.drag = drag

		# System state space
		self.pos = np.zeros((3,))
		self.vel = np.zeros((3,))
		self.angle = np.array([1., 0., 0., 0.])  # Quaternion format: qw, qx, qy, qz
		self.a_rate = np.zeros((3,))

		# Input constraints
		self.max_input_value = 1  # Motors at full thrust
		self.min_input_value = 0  # Motors turned off

		# Quadrotor intrinsic parameters
		self.J = np.array([.03, .03, .06])  # N m s^2 = kg m^2
		self.mass = 1.0  # kg

		# Length of motor to CoG segment
		self.length = 0.47 / 2  # m

		# Positions of thrusters

		self.x_f = np.array([self.length, 0, -self.length, 0])
		self.y_f = np.array([0, self.length, 0, -self.length])

		# For z thrust torque calculation
		self.c = 0.013  # m   (z torque generated by each motor)
		self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

		# Gravity vector
		self.g = np.array([[0], [0], [9.81]])  # m s^-2

		# Actuation thrusts
		self.u = np.array([0.0, 0.0, 0.0, 0.0])  # N

		# Drag coefficients [kg / m]
		self.rotor_drag_xy = 0.3
		#self.rotor_drag_xy = 0.0
		#self.rotor_drag_xy = 30
		self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
		#self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
		
		self.rotor_drag = np.array([self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z])[:, np.newaxis]
		self.aero_drag = 0.08
		#self.aero_drag = 80

		self.payload_mass = 0.3  # kg
		self.payload_mass = self.payload_mass * payload


	def set_state(self, *args):
		assert len(args) == 1 and len(args[0]) == 13
		self.pos[0], self.pos[1], self.pos[2], \
		self.angle[0], self.angle[1], self.angle[2], self.angle[3], \
		self.vel[0], self.vel[1], self.vel[2], \
		self.a_rate[0], self.a_rate[1], self.a_rate[2] \
			= args[0]


	def get_state(self, quaternion=False, stacked=False, body_frame=False):


		if body_frame:
			v_b = v_dot_q(self.vel, quaternion_inverse(self.angle)) # body frame velocity

			if quaternion and not stacked:
				return [self.pos, self.angle, v_b, self.a_rate]

			if quaternion and stacked:
				return [self.pos[0], self.pos[1], self.pos[2], self.angle[0], self.angle[1], self.angle[2], self.angle[3],
				v_b[0], v_b[1], v_b[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]]

		else:


			if quaternion and not stacked:
				return [self.pos, self.angle, self.vel, self.a_rate]

			if quaternion and stacked:
				return [self.pos[0], self.pos[1], self.pos[2], self.angle[0], self.angle[1], self.angle[2], self.angle[3],
						self.vel[0], self.vel[1], self.vel[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]]

			angle = quaternion_to_euler(self.angle) # Euler angles
			if not quaternion and stacked :
				return [self.pos[0], self.pos[1], self.pos[2], angle[0], angle[1], angle[2],
					self.vel[0], self.vel[1], self.vel[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]]

			if not quaternion and not stacked:
				return [self.pos, angle, self.vel, self.a_rate]

		


		






	def get_control(self):

		return self.u

	def update(self, u, dt):
		"""
		Runge-Kutta 4th order dynamics integration

		:param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
		:param dt: time differential
		"""
		f_d = np.zeros((3, 1))
		t_d = np.zeros((3, 1))

		self.u = u

		x = self.get_state(quaternion=True, stacked=False)
		#print('u: ' + str(self.u))
		#print('dv: ' + str(self.f_vel(x, self.u, f_d)))
		# RK4 integration
		k1 = [self.f_pos(x), self.f_att(x), self.f_vel(x, self.u, f_d), self.f_rate(x, self.u, t_d)]
		x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
		k2 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
		x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
		k3 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
		x_aux = [x[i] + dt * k3[i] for i in range(4)]
		k4 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
		x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in
		     range(4)]


		# Ensure unit quaternion
		x[1] = unit_quat(x[1])

		self.pos, self.angle, self.vel, self.a_rate = x
		#print(self.pos)


	def get_aero_drag(self, x, body_frame=False):
		"""
		Aerodynamic drag affecting the quad
		:param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
		:return: acceleration vector length 3
		"""
		if self.drag:
			# Transform velocity to body frame
			v_b = v_dot_q(x[2], quaternion_inverse(x[1]))[:, np.newaxis]
			# Compute aerodynamic drag acceleration in world frame
			a_drag = -self.aero_drag * v_b ** 2 * np.sign(v_b) / self.mass
			# Add rotor drag
			a_drag -= self.rotor_drag * v_b / self.mass
			if not body_frame:
				# Transform drag acceleration to world frame
				a_drag = v_dot_q(a_drag, x[1])
		else:
			a_drag = np.zeros((3, 1))
		return a_drag


	def f_pos(self, x):
		"""
		Time-derivative of the position vector
		:param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
		:return: position differential increment (vector): d[pos_x; pos_y]/dt
		"""

		vel = x[2]
		return vel

	def f_att(self, x):
		"""
		Time-derivative of the attitude in quaternion form
		:param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
		:return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
		"""

		rate = x[3]
		angle_quaternion = x[1]

		return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

	def f_vel(self, x, u, f_d):
		"""
		Time-derivative of the velocity vector
		:param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
		:param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
		:param f_d: disturbance force vector (3-dimensional)
		:return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
		"""

		f_thrust = u * self.max_thrust
		a_thrust = np.array([[0], [0], [np.sum(f_thrust)]]) / self.mass



		a_drag = self.get_aero_drag(x)
		#a_drag = np.zeros((3, 1))

		angle_quaternion = x[1]

		a_payload = -self.payload_mass * self.g / self.mass

		return np.squeeze(-self.g + a_payload + a_drag + v_dot_q(a_thrust + f_d / self.mass, angle_quaternion))

	def f_rate(self, x, u, t_d):
		"""
		Time-derivative of the angular rate
		:param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
		:param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
		:param t_d: disturbance torque (3D)
		:return: angular rate differential increment (scalar): dr/dt
		"""
		f_thrust = u*self.max_thrust
		rate = x[3]
		return np.array([
			1 / self.J[0] * (f_thrust.dot(self.y_f) + t_d[0] + (self.J[1] - self.J[2]) * rate[1] * rate[2]),
			1 / self.J[1] * (-f_thrust.dot(self.x_f) + t_d[1] + (self.J[2] - self.J[0]) * rate[2] * rate[0]),
			1 / self.J[2] * (f_thrust.dot(self.z_l_tau) + t_d[2] + (self.J[0] - self.J[1]) * rate[0] * rate[1])
		]).squeeze()
