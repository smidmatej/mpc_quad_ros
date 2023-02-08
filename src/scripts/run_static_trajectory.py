import os



dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level


static_result_filename = os.path.join(dir_path, '..', 'outputs/python_simulation/data/static_dataset.pkl')
os.system('python ../execute_trajectory.py -o ' + static_result_filename + ' --gpe 0' + \
            ' --trajectory 0 --v_max 25 --a_max 20 --show 1')



