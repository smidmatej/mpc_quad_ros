import os
import sys
sys.path.append('../')
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level
environment = 'python_simulation'
data_filename = 'traj2_v10_a10_gp0'


trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
result_plot_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', '3d_' + environment + '_' + data_filename + '.pdf')

visualiser = Visualiser(trajectory_filename)
visualiser.create_3D_plot(result_plot_filename)