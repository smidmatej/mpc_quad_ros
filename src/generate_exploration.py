import os


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    
    simulation_result_fname = os.path.join(dir_path, '..', 'outputs/python_simulation/data/exploration_dataset')
    os.system('python explore_trajectories.py -o ' + simulation_result_fname + \
                ' --trajectory 1 --show 1')

    

if __name__ == '__main__':
    main()