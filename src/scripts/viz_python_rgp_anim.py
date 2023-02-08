import os
import sys
sys.path.append('../')
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level
environment = 'python_simulation'
data_filename = 'trajectory_v15_a5_gp2'

trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
result_animation_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', 'anim_' + environment + '_' + data_filename)


assert "gp2" in data_filename, "data_filename must contain 'gp2' to be a training data file."

visualiser = Visualiser(trajectory_filename)
visualiser.create_rgp_animation(result_animation_filename, 100, True)
