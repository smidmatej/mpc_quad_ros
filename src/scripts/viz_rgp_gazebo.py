import os
import sys
sys.path.append('../')
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level
environment = 'gazebo_simulation'
data_filename = 'training_v15_a5_gp2'

trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
result_animation_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', 'rgp_full_anim_' + environment + '_' + data_filename)

#data_filename must contain "gp2" to be a training data file. This is a hack to ensure that the correct data file is used.
assert "gp2" in data_filename, "data_filename must contain 'gp2' to be a training data file."

visualiser = Visualiser(trajectory_filename)
visualiser.create_rgp_full_animation(result_animation_filename, 10, True)
