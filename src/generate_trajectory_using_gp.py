import os


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    
    simulation_result_fname = os.path.join(dir_path, '..', 'outputs/python_simulation/data/trajectory_using_gp.pkl')
    os.system('python execute_trajectory.py -o ' + simulation_result_fname + ' --gpe 1' + \
                ' --trajectory 1 --v_max 10 --a_max 10 --show 1')

    

if __name__ == '__main__':
    main()