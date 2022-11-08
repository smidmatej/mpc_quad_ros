
import numpy as np
import pyquaternion

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
	
	q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
	yaw, pitch, roll = q.yaw_pitch_roll
	return [roll, pitch, yaw]

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    return np.array([[0, -v[0], -v[1], -v[2]],
                     [v[0], 0, v[2], -v[1]],
                     [v[1], -v[2], 0, v[0]],
                     [v[2], v[1], -v[0], 0]])

def unit_quat(q):
	"""
	Normalizes a quaternion to be unit modulus.
	:param q: 4-dimensional numpy array
	:return: the unit quaternion in the same data format as the original one
	"""

	#breakpoint()
	q_norm = np.sqrt(np.sum(q ** 2))
	return q/q_norm

def v_dot_q(v, q):
	rot_mat = q_to_rot_mat(q)

	return rot_mat.dot(v)

def q_to_rot_mat(q):
	
	qw, qx, qy, qz = q[0], q[1], q[2], q[3]
	rot_mat = np.array([
		[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
		[2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
		[2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

	return rot_mat