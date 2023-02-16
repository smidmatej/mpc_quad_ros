import os
import sys
sys.path.append('../')
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level
environment = 'python_simulation'
data_filename = 'traj0_v10_a10_gp2'

trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
result_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', data_filename)


data_path = os.path.join(dir_path, 'covariance_data.csv')
visualiser = Visualiser(trajectory_filename)
visualiser.visualize_cov_data(data_path, result_filename)
