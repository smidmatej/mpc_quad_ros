import os


def main():

    simulation_result_fname = 'data/training_dataset.pkl'
    os.system('python execute_trajectory.py -o ' + simulation_result_fname + ' --gpe 0' + \
                ' --trajectory 1 --v_max 25 --a_max 20 --show 1')

    

if __name__ == '__main__':
    main()