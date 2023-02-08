import os



dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level


simulation_result_fname = os.path.join(dir_path, '..', 'outputs/python_simulation/data/trajectory_using_gp.pkl')
os.system('python execute_trajectory.py -o ' + simulation_result_fname + ' --gpe 1' + \
            ' --trajectory 1 --v_max 10 --a_max 10 --show 1')


