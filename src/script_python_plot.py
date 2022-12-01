import os
from Visualiser import Visualiser
dir_path = os.path.dirname(os.path.realpath(__file__))
simulation = 'python_simulation'


trajectory_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'data', 'static_dataset.pkl')
result_animation_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'animations', 'animation.mp4')
result_plot_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'img', 'trajectory.pdf')

visualiser = Visualiser(trajectory_filename)

#visualiser.create_animation(result_animation_filename, 100, True)
visualiser.plot_data(result_plot_filename)