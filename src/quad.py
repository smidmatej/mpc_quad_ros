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

from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse, parse_xacro_file



class Quadrotor3D:

	def __init__(self, payload : bool = False, drag : bool = False):
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
		self.g = np.array([0, 0, 9.81])  # m s^-2

		# Actuation thrusts
		self.u = np.array([0.0, 0.0, 0.0, 0.0])  # N

		# Drag coefficients [kg / m]
		self.rotor_drag_xy = 0.3
		#self.rotor_drag_xy = 0.0
		#self.rotor_drag_xy = 30
		self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
		#self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
		
		self.rotor_drag = np.array([self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z])
		#self.aero_drag = 0.08
		self.aero_drag = 0.8

		self.payload_mass = 0.3  # kg
		self.payload_mass = self.payload_mass * payload




	def set_state(self, x : np.ndarray):
		"""
		Set the state of the quadrotor
		:param x: 13-element state vector
		"""
		self.pos = x[0:3]
		self.angle = x[3:7]
		self.vel = x[7:10]
		self.a_rate = x[10:13]


	def get_state(self, quaternion : bool = False, stacked : bool = False, body_frame : bool = False) -> 'np.ndarray | list':
		"""
		Get the state of the quadrotor
		:param quaternion: Whether to return the state in quaternion format or in Euler angles
		:param stacked: Whether to return the state as a 13-element vector or as a 4-element list of vectors
		:param body_frame: Whether to return the velocity in the body frame or in the inertial frame
		:return: The state of the quadrotor as a 13-element vector if stacked or as a 4-element list of vectors if not stacked
		"""

		if body_frame:
			v_b = v_dot_q(self.vel, quaternion_inverse(self.angle)) # body frame velocity
			if quaternion:
				if stacked:
					return np.array([self.pos[0], self.pos[1], self.pos[2], self.angle[0], self.angle[1], self.angle[2], self.angle[3],
						v_b[0], v_b[1], v_b[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]])
				else: # Not stacked
					return [self.pos, self.angle, v_b, self.a_rate]

			else: # Not quaternion
				angle = quaternion_to_euler(self.angle) # Euler angles
				if stacked:
					return np.array([self.pos[0], self.pos[1], self.pos[2], angle[0], angle[1], angle[2],
						v_b[0], v_b[1], v_b[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]])

				else: # Not stacked
					return [self.pos, angle, v_b, self.a_rate]

		else: # Not body frame
			if quaternion:
				if stacked:
					return np.array([self.pos[0], self.pos[1], self.pos[2], self.angle[0], self.angle[1], self.angle[2], self.angle[3],
						self.vel[0], self.vel[1], self.vel[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]])

				else:
					return [self.pos, self.angle, self.vel, self.a_rate]
			else: # Not quaternion
				angle = quaternion_to_euler(self.angle) # Euler angles
				if stacked:
					return np.array([self.pos[0], self.pos[1], self.pos[2], angle[0], angle[1], angle[2],
						self.vel[0], self.vel[1], self.vel[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]])

				else: # Not stacked	
					return [self.pos, angle, self.vel, self.a_rate]
			

		

	def get_control(self):
		"""
		Get the control input of the quadrotor
		:return: 4-element control input vector
		"""
		return self.u


	def one_step_forward(self, x : np.ndarray, u : np.ndarray, dt : float, f_d : np.ndarray = np.zeros((3,)), t_d : np.ndarray = np.zeros((3,))) -> np.ndarray:
		"""
		Runge-Kutta 4th order dynamics integration
		:param x: 13-dimensional state vector
		:param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
		:param dt: time differential
		:param f_d: 3-dimensional disturbance force vector
		:param t_d: 3-dimensional disturbance torque vector
		"""
		assert x.shape == (13,)
		assert u.shape == (4,)
		assert f_d.shape == (3,)
		assert t_d.shape == (3,)
		assert np.all(u >= 0.0) and np.all(u <= 1.0)

		# ---- RK4 integration ----
		k1 = self.f_nominal(x, u, f_d, t_d)
		k2 = self.f_nominal(x + dt / 2 * k1, u, f_d, t_d)
		k3 = self.f_nominal(x + dt / 2 * k2, u, f_d, t_d)
		k4 = self.f_nominal(x + dt * k3, u, f_d, t_d)
		x_out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

		#x_out[3:7] = unit_quat(x_out[3:7]) # Normalize quaternion

		return x_out


	'''
	def one_step_forward_predict(self, x : 'np.ndarray | list', u : np.ndarray, dt : float, f_d : np.ndarray = np.zeros((3,)), t_d : np.ndarray = np.zeros((3,))) -> np.ndarray:
		"""
		Runge-Kutta 4th order dynamics integration
		:param x: 13-dimensional state vector
		:param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
		:param dt: time differential
		:param f_d: 3-dimensional disturbance force vector
		:param t_d: 3-dimensional disturbance torque vector
		"""
		assert type(x) == np.ndarray or type(x) == list
		
		if type(x) == np.ndarray:
			# Unpack state vector into a list
			x_list = [None]*4
			x_list[0] = x[0:3]
			x_list[1] = x[3:7]
			x_list[2] = x[7:10]
			x_list[3] = x[10:13]
			x = x_list

		# RK4 integration
		k1 = np.concatenate([self.f_pos(x), self.f_att(x), self.f_vel(x, self.u, f_d), self.f_rate(x, self.u, t_d)])
		x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
		k2 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
		x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
		k3 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
		x_aux = [x[i] + dt * k3[i] for i in range(4)]
		k4 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
		x_out = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in
		     range(4)]

		# Ensure unit quaternion
		x_out[1] = unit_quat(x_out[1])

		x_out = np.array([x_out[0], x_out[1], x_out[2], x_out[3], x_out[4], x_out[5], x_out[6],
		x_out[0], x_out[1], x_out[2], x_out[10], x_out[11], x_out[12]])

		return x_out		
	'''

	def update(self, u : np.ndarray, dt : float):
		"""
		Runs the one_step_forward_predict function and from the current state updates the state of the quadrotor
		:param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
		:param dt: time differential
		"""
		assert u.shape == (4,)
		assert np.all(u >= 0.0) and np.all(u <= 1.0)

		self.u = u # Update control input
		x = self.get_state(quaternion=True, stacked=True) # Get current state

		x_next = self.one_step_forward(x, u, dt) # Runge-Kutta 4th order integration
		self.set_state(x_next)


	def get_aero_drag(self, x : np.ndarray, body_frame : bool = False):
		"""
		Aerodynamic drag affecting the quad
		:param x: 13-length array state vector
		:return: 3-length acceleration vector 
		"""
		assert x.shape == (13,)
		if self.drag:
			# Transform velocity to body frame
			v_b = v_dot_q(x[7:10], quaternion_inverse(x[3:7]))
			# Compute aerodynamic drag acceleration in world frame
			a_drag = -self.aero_drag * v_b ** 2 * np.sign(v_b) / self.mass
			# Add rotor drag
			a_drag -= self.rotor_drag * v_b / self.mass

			if not body_frame:
				# Transform drag acceleration to world frame
				a_drag = v_dot_q(a_drag, x[3:7])
		else:
			a_drag = np.zeros((3, 1))

		return a_drag


	def f_nominal(self, x : np.ndarray, u : np.ndarray, f_d : np.ndarray = np.zeros((3,)), t_d : np.ndarray = np.zeros((3,))) -> np.ndarray:
		"""
		Computes the nominal dynamics of the quadrotor
		:param x: 13-dimensional state vector
		:param u: 4-dimensional control vector
		:param f_d: 3-dimensional disturbance force vector
		:param t_d: 3-dimensional disturbance torque vector
		:return: 13-dimensional state derivative vector
		"""
		assert x.shape == (13, ), 'x must be a 13-length array'
		assert u.shape == (4, ), 'u must be a 4-length array'
		assert f_d.shape == (3, ), 'f_d must be a 3-length array'
		assert t_d.shape == (3, ), 't_d must be a 4-length array'

		dpos = self.f_pos(x)
		datt = self.f_att(x)
		dvel = self.f_vel(x, u, f_d)
		drate = self.f_rate(x, u, t_d)

		dx = np.concatenate([dpos, datt, dvel, drate])
		assert dx.shape == (13, ), 'dx must be a 13-length array'
		return dx



	def f_pos(self, x : np.ndarray) -> np.ndarray:
		"""
		Time-derivative of the position vector
		:param x: 13-length state array
		:return: position differential increment (vector): d[pos_x; pos_y]/dt
		"""
		assert x.shape == (13, ), 'x must be a 13-length array'
		vel = x[7:10]
		return vel

	def f_att(self, x : np.ndarray) -> np.ndarray:
		"""
		Time-derivative of the attitude in quaternion form
		:param x: 13-length state array
		:return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
		"""
		assert x.shape == (13, ), 'x must be a 13-length array'
		rate = x[10:13]
		angle_quaternion = x[3:7]

		dq = 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)
		assert dq.shape == (4, ), 'dq must be a 4-length array'
		return dq

	def f_vel(self, x : np.ndarray, u : np.ndarray , f_d : np.ndarray) -> np.ndarray:
		"""
		Time-derivative of the velocity vector
		:param x: 13-length state array
		:param u: control input array (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
		:param f_d: disturbance force vector (3-dimensional)
		:return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
		"""
		assert x.shape == (13, ), 'x must be a 13-length array'
		assert u.shape == (4, ), 'u must be a 4-length array'
		assert f_d.shape == (3, ), 'f_d must be a 3-length array'

		angle_quaternion = x[3:7]

		f_thrust = u * self.max_thrust
		a_thrust_body = np.array([0.0, 0.0, np.sum(f_thrust)]) / self.mass # Thrust acceleration in body frame
		a_thrust_world = v_dot_q(a_thrust_body, angle_quaternion)

		a_d_body = f_d / self.mass # Disturbance acceleration in body frame
		a_d_world = v_dot_q(a_d_body, angle_quaternion)

		a_drag_world = self.get_aero_drag(x, body_frame=False) # Aerodynamic drag acceleration in world frame
		
		a_payload = -self.payload_mass * self.g / self.mass # TODO : This is bullshit. The quad experiences the same acceleration regardless of the payload mass

		dvel = -self.g + a_payload + a_drag_world + a_thrust_world + a_d_world
		assert dvel.shape == (3, ), 'dvel must be a 3-length array'
		return dvel

	def f_rate(self, x : np.ndarray, u : np.ndarray, t_d : np.ndarray) -> np.ndarray:
		"""
		Time-derivative of the angular rate
		:param x: 13-length state array
		:param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
		:param t_d: disturbance torque (3D)
		:return: angular rate differential increment (scalar): dr/dt
		"""
		assert x.shape == (13, ), 'x must be a 13-length array'
		assert u.shape == (4, ), 'u must be a 4-length array'
		assert t_d.shape == (3, ), 'f_d must be a 3-length array'

		f_thrust = u*self.max_thrust
		rate = x[10:13]

		drate = np.array([
			1 / self.J[0] * (f_thrust.dot(self.y_f) + t_d[0] + (self.J[1] - self.J[2]) * rate[1] * rate[2]),
			1 / self.J[1] * (-f_thrust.dot(self.x_f) + t_d[1] + (self.J[2] - self.J[0]) * rate[2] * rate[0]),
			1 / self.J[2] * (f_thrust.dot(self.z_l_tau) + t_d[2] + (self.J[0] - self.J[1]) * rate[0] * rate[1])
		]).squeeze()

		assert drate.shape == (3, ), 'drate must be a 3-length array'
		return drate


	
	def set_parameters_from_file(self, params_filepath : str, quad_name):
		"""
		Sets the parameters of this quad to those from the provided xarco file
		:param params_filepath: path to the xacro file to load parameters from
		:param quad_name: name of the quad in the xacro files
		"""

		# Get parameters for drone
		attrib = parse_xacro_file(params_filepath)

		self.mass = float(attrib['mass']) + float(attrib['mass_rotor']) * 4

		self.J = np.array([float(attrib['body_inertia'][0]['ixx']),
						float(attrib['body_inertia'][0]['iyy']),
						float(attrib['body_inertia'][0]['izz'])])
		self.length = float(attrib['arm_length'])

		# Max thrust of 1 rotor
		self.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
		self.c = float(attrib['moment_constant'])

		# x configuration
		if quad_name != "hummingbird":
			h = np.cos(np.pi / 4) * self.length
			self.x_f = np.array([h, -h, -h, h])
			self.y_f = np.array([-h, -h, h, h])
			self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

		# + configuration
		else:
			self.x_f = np.array([self.length, 0, -self.length, 0])
			self.y_f = np.array([0, self.length, 0, -self.length])
			self.z_l_tau = -np.array([-self.c, self.c, -self.c, self.c])
