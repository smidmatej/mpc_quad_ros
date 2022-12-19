import os
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
environment = 'gazebo_simulation'
data_filename = 'test_circle_v20_a5_gp1'
data_filename = 'test_circle_v20_a5_gp0'

trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
result_plot_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', 'img', environment,  'plot_' + data_filename + '.pdf')

visualiser = Visualiser(trajectory_filename)
visualiser.plot_data(result_plot_filename)
