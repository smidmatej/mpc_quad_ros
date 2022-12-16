import os
from Visualiser import Visualiser
dir_path = os.path.dirname(os.path.realpath(__file__))
simulation = 'gazebo_simulation'


trajectory_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'data', 'random_trajectory_nogp.pkl')
result_plot_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'img', 'trajectory.pdf')

visualiser = Visualiser(trajectory_filename)

visualiser.plot_data(result_plot_filename)