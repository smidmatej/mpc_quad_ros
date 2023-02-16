import os
import sys
sys.path.append('../')
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level

data_filename = 'trajectory_sampled'

data_path = os.path.join(dir_path, 'trajectory_generation', 'trajectories', data_filename + '.csv')
result_filepath = os.path.join(dir_path, '..', 'outputs', 'graphics', data_filename + '.pdf')


visualiser = Visualiser()
visualiser.visualize_trajectory(data_path, result_filepath)
