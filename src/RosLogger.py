import numpy as np

import os 
from utils.save_dataset import save_dict
import rospy

class RosLogger:
    def __init__(self, filename) -> None:
        
        self.dictionary = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.filename_npy = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', filename + '.npy')
        self.filename_dict = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', filename + '.pkl')
        self.filename_csv = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', filename + '.csv')


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
        #pd.DataFrame(self.dictionary['odom']).to_csv(self.filename_csv, header=csv_header, index=None)

        # dictinary to numpy array
        output_dict = {}
        for key in self.dictionary:
            output_dict[key] = np.array(self.dictionary[key])

        rospy.loginfo(f"Saving trajectory to {self.filename_dict}")
        save_dict(output_dict, self.filename_dict)
        #np.save(self.filename_npy, self.dictionary['odom'])