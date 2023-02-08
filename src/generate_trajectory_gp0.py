import os


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    vmax = 15
    amax = 5
    trajectory = 1
    gpe = 0
    simulation_result_fname = os.path.join(dir_path, '..', f'outputs/python_simulation/data/trajectory_v{vmax}_a{amax}_gp{gpe}')
    os.system('python execute_trajectory.py -o ' + simulation_result_fname + f' --gpe {gpe}' + \
                f' --trajectory {trajectory} --v_max {vmax} --a_max {amax} --show 1')

    

if __name__ == '__main__':
    main()