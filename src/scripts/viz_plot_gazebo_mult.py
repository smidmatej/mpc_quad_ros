import os
import sys
sys.path.append('../')
from Visualiser import Visualiser

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level
environment = 'gazebo_simulation'

filenames = ['traj0_v3_a3_gp1','traj0_v6_a6_gp1','traj0_v9_a9_gp1','traj0_v12_a12_gp1', 'traj2_v3_a3_gp1', 'traj2_v6_a6_gp1', 'traj2_v9_a9_gp1', 'traj2_v12_a12_gp1']
for data_filename in filenames:
    trajectory_filename = os.path.join(dir_path, '..', 'outputs', environment, 'data', data_filename + '.pkl')
    result_plot_filename = os.path.join(dir_path, '..', 'outputs', 'graphics', 'plot_' + environment + '_' + data_filename + '.pdf')

    visualiser = Visualiser(trajectory_filename)
    visualiser.plot_data(result_plot_filename)