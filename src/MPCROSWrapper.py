#!/usr/bin/env python3
import rospy


from geometry_msgs.msg import Pose
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry
import std_msgs
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))



from source.quad import Quadrotor3D
from source.quad_opt import quad_optimizer
from source.utils.utils import load_trajectory, get_reference_chunk, parse_xacro_file


class MPCROSWrapper:
        
    def __init__(self, quad_name='hummingbird'):


        self.quad_name = quad_name
        # MPC prediction horizon
        t_lookahead = 1 # Prediction horizon duration
        n_nodes = 10 # Prediction horizon number of timesteps in t_lookahead


        # Instantiate quadrotor model with default parameters
        self.quad = Quadrotor3D(payload=False, drag=True) # Controlled plant s
        # Loads parameters of  a quad from a xarco file into quad object
        self.quad = set_quad_parameters_from_file(self.quad, self.quad_name)
        # Creates an optimizer object for the quad
        self.quad_opt = quad_optimizer(self.quad, t_horizon=t_lookahead, n_nodes=n_nodes, gpe=None) # computing optimal control over model of plant
        

        


def set_quad_parameters_from_file(quad, quad_name):

    this_path = os.path.dirname(os.path.realpath(__file__))
    print(f'this_path: {this_path}')
    params_filename = os.path.join(this_path, '..' , 'config', quad_name + '.xacro')
    print(f'params_filename: {params_filename}')

    # Get parameters for drone
    attrib = parse_xacro_file(params_filename)

    quad.mass = float(attrib['mass']) + float(attrib['mass_rotor']) * 4
    quad.J = np.array([float(attrib['body_inertia'][0]['ixx']),
                    float(attrib['body_inertia'][0]['iyy']),
                    float(attrib['body_inertia'][0]['izz'])])
    quad.length = float(attrib['arm_length'])

    quad.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
    quad.c = float(attrib['moment_constant'])

    # x configuration
    if quad_name != "hummingbird":
        h = np.cos(np.pi / 4) * quad.length
        quad.x_f = np.array([h, -h, -h, h])
        quad.y_f = np.array([-h, -h, h, h])
        quad.z_l_tau = np.array([-quad.c, quad.c, -quad.c, quad.c])

    # + configuration
    else:
        quad.x_f = np.array([quad.length, 0, -quad.length, 0])
        quad.y_f = np.array([0, quad.length, 0, -quad.length])
        quad.z_l_tau = -np.array([-quad.c, quad.c, -quad.c, quad.c])

    return quad





def main():    
    mpc_ros_wrapper = MPCROSWrapper()
    print("controller initialized")
    rospy.spin()   
if __name__ == '__main__':
    main()
