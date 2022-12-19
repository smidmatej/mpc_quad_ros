import os
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
environment = 'gazebo_simulation'
data_filename = 'test_circle_v20_a5_gp0'

trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
result_animation_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', 'anim_' + environment + '_' + data_filename + '.mp4')

visualiser = Visualiser(trajectory_filename)
visualiser.create_animation(result_animation_filename, 100, True)
