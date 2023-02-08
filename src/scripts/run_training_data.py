import os



dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level


simulation_result_fname = os.path.join(dir_path, '..', 'outputs/python_simulation/data/training_dataset.pkl')
os.system('python ../execute_trajectory.py -o ' + simulation_result_fname + ' --gpe 0' + \
            ' --trajectory 1 --v_max 25 --a_max 20 --show 1')



