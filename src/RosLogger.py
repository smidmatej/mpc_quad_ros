import numpy as np

import os 
from utils.save_dataset import save_dict


class RosLogger:
    def __init__(self) -> None:
        
        self.dictionary = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.filename_npy = dir_path + '../outputs/log.npy'
        self.filename_dict = dir_path + '../outputs/log.pkl'
        self.filename_csv = dir_path + '../outputs/log.csv'


    def log(self, input_dict):
        for key in input_dict:
            if key not in self.dictionary:
                self.dictionary[key] = list()
                self.dictionary[key].append(input_dict[key])
            else:
                #print(type(self.dictionary[key]))
                self.dictionary[key].append(input_dict[key])


    def save_log(self):

        #csv_header = ['time', 'x', 'y', 'z', 'qw', 'qx', 'qy' 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        #pd.DataFrame(self.dictionary['odom']).to_csv(self.filename_csv, header=csv_header, index=None)

        # dictinary to numpy array
        output_dict = {}
        for key in self.dictionary:
            output_dict[key] = np.array(self.dictionary[key])

        save_dict(output_dict, self.filename_dict)
        #np.save(self.filename_npy, self.dictionary['odom'])