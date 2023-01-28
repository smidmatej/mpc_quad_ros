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

import os 
from utils.save_dataset import save_dict
import rospy

class Logger:
    def __init__(self, filepath) -> None:
        
        self.dictionary = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.filepath_npy = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', filepath + '.npy')
        self.filepath_dict = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', filepath + '.pkl')
        self.filepath_csv = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', filepath + '.csv')

    def clear_memory(self):
        self.dictionary = {}

    def log(self, input_dict):
        for key in input_dict:
            if key not in self.dictionary:
                self.dictionary[key] = list()
                self.dictionary[key].append(input_dict[key])
            else:
                #rospy.loginfo(type(self.dictionary[key]))
                self.dictionary[key].append(input_dict[key])


    def save_log(self):

        #csv_header = ['time', 'x', 'y', 'z', 'qw', 'qx', 'qy' 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        #pd.DataFrame(self.dictionary['odom']).to_csv(self.filepath_csv, header=csv_header, index=None)

        # dictinary to numpy array
        output_dict = {}
        for key in self.dictionary:
            output_dict[key] = np.array(self.dictionary[key])

        rospy.loginfo(f"Saving trajectory to {self.filepath_dict}")
        print(f"Saving trajectory to {self.filepath_dict}")
        save_dict(output_dict, self.filepath_dict)
        #np.save(self.filepath_npy, self.dictionary['odom'])