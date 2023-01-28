#!/usr/bin/env python3

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


import rospy
import numpy as np
import os 

from quad import Quadrotor3D
from quad_opt import quad_optimizer
from utils.utils import parse_xacro_file
from gp.GPE import GPEnsemble

class MPCROSWrapper:
        
    def __init__(self, quad_name='hummingbird', use_gp=False):


        self.quad_name = quad_name
        self.use_gp = use_gp
        # MPC prediction horizon
        self.t_lookahead = 1.0 # Prediction horizon duration
        self.n_nodes = 10 # Prediction horizon number of timesteps in self.t_lookahead


        # Instantiate quadrotor model with default parameters
        self.quad = Quadrotor3D(payload=False, drag=True) # Controlled plant s
        # Loads parameters of  a quad from a xarco file into quad object
        self.quad = set_quad_parameters_from_file(self.quad, self.quad_name)


        dir_path = os.path.dirname(os.path.realpath(__file__))
        gp_path = rospy.get_param('/mpcros/mpc_controller/gp_path')
        self.ensemble_path = os.path.join(dir_path, gp_path)

        self.initialize()


    def initialize(self):
        """
        Initialize the quad optimizer with the quadrotor model and the GP ensemble.
        This method exists to be called inside the MPCROSWrapper constructor and also to be called when the mpc_controller_node retrains the GP ensemble.
        """

        if self.use_gp == 0:
            print("Not using GPE")
            gpe = None
        elif self.use_gp == 1:
            gpe = GPEnsemble.fromdir(self.ensemble_path, "GP")
        elif self.use_gp == 2:
            gpe = GPEnsemble.fromdir(self.ensemble_path, "RGP")
        else:
            raise ValueError("Invalid GPE argument")

        # Creates an optimizer object for the quad
        self.quad_opt = quad_optimizer(self.quad, t_horizon=self.t_lookahead, n_nodes=self.n_nodes, gpe=gpe) # computing optimal control over model of plant
        
        


def set_quad_parameters_from_file(quad, quad_name):
    """
    Loads parameters of a quad from a xarco file into quad object
    :param quad: Quadrotor3D object to load parameters into
    :param quad_name: name of quadrotor to load parameters for (e.g. 'hummingbird'). Concatenate with '.xacro' to get filename
    """
    
    this_path = os.path.dirname(os.path.realpath(__file__))
    rospy.loginfo(f'this_path: {this_path}')
    params_filename = os.path.join(this_path, '..' , 'config', quad_name + '.xacro')
    rospy.loginfo(f'params_filename: {params_filename}')

    #params_filename = os.path.join('..' , 'config', quad_name + '.xacro')


    # Get parameters for drone
    attrib = parse_xacro_file(params_filename)

    quad.mass = float(attrib['mass']) + float(attrib['mass_rotor']) * 4




    quad.J = np.array([float(attrib['body_inertia'][0]['ixx']),
                    float(attrib['body_inertia'][0]['iyy']),
                    float(attrib['body_inertia'][0]['izz'])])
    quad.length = float(attrib['arm_length'])

    # Max thrust of 1 rotor
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
    rospy.loginfo("controller initialized")
    rospy.spin()   

    
if __name__ == '__main__':
    main()
