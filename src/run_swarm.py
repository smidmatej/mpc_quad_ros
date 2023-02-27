import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, '..') # go up one level

vmax = 1
amax = 1
trajectory = 2
gpe = 0
simulation_result_fname = os.path.join(dir_path, '..', f'outputs/python_simulation/data/trajectory_v{vmax}_a{amax}_gp{gpe}')
os.system('python cf_execute.py -o ' + simulation_result_fname + f' --gpe {gpe}' + \
            f' --trajectory {trajectory} --v_max {vmax} --a_max {amax} --show 1')



