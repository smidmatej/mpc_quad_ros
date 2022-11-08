from utils.save_dataset import load_dict
import matplotlib.pyplot as plt
import numpy as np
import argparse 

import os
import json

from tqdm.contrib.telegram import tqdm, trange



def main():

    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument

    parser.add_argument("-n", "--new_data", type=int, required=False, default=1, help="Generate new data")

    # Read arguments from command line
    args = parser.parse_args()

    compare_config_file = 'comparisson_config.json'
    with open(compare_config_file) as json_file:
        config = json.load(json_file)

    
    keys = ['gpe', 'no_gpe']
    mean_rmse_pos = dict.fromkeys(keys)
    v_max = dict.fromkeys(keys)
    for key in keys:
        mean_rmse_pos[key] = [None]*len(config['runs'])
        v_max[key] = [None]*len(config['runs'])

    sim_result_dict = [None]*len(config['runs'])



    for n, run in zip(range(len(config['runs'])), config['runs']):

        simulation_result_fname = 'data/sim_' + str(run['gpe']) + '_trajectory' + str(run['trajectory']) + \
                                    '_v_max' + str(run['v_max']) + '_a_max' + str(run['a_max']) + '.pkl'

        plot_result_fname = 'img/traj_' + str(run['gpe']) + '_trajectory' + str(run['trajectory']) + \
                            '_v_max' + str(run['v_max']) + '_a_max' + str(run['a_max']) + '.pdf'

        if bool(args.new_data):
            print('Generating new data')
            os.system('python execute_trajectory.py -o ' + simulation_result_fname + ' --gpe ' + str(run['gpe']) + \
                        ' --trajectory ' + str(run['trajectory']) + ' --v_max ' + str(run['v_max']) \
                            + ' --a_max '+ str(run['a_max']) + ' --show 0' + ' --plot_filename ' + plot_result_fname)

        else:
            print('Loading old data')

        
        sim_result_dict[n] = load_dict(simulation_result_fname)

        # leave out the last second of the simulation, because it tries to stop in place
        n_drop = int(1/sim_result_dict[n]['dt'])
        print(f'Dropping last {n_drop} samples')
        v_norm = np.linalg.norm(sim_result_dict[n]['v'][:-n_drop,:], axis=1)


        connect_dict = {'gpe':1, 'no_gpe':0}
        for key in keys:
            if connect_dict[key] == run['gpe']:
                v_max[key][n] = np.max(v_norm)
                mean_rmse_pos[key][n] = np.mean(sim_result_dict[n]['rmse_pos'][:-n_drop])





        #print(f'v_max: {v_max}')
        #print(f'mean_rmse_pos: {mean_rmse_pos}')


    
    colors = {keys[0]: 'r', keys[1]: 'b'}


    colors = {keys[0]: 'r', \
            keys[1] : 'b'} 

    plt.figure(figsize=(10,6), dpi=100)
    for idx_k, key in zip(range(len(keys)), keys):

        filtered_v_max = [i for i in v_max[key] if i is not None]
        filtered_mean_rmse_pos = [i for i in mean_rmse_pos[key] if i is not None]

        plt.scatter(filtered_v_max, filtered_mean_rmse_pos, c=colors[key], label=key)
        plt.plot(filtered_v_max, filtered_mean_rmse_pos, '--', linewidth=0.8, c=colors[key], label=key)

        
    plt.legend()
    plt.xlabel('v_max [m/s]')
    plt.ylabel('mean rmse pos [m]')

    plot_filename = "img/gpe_comparison.pdf"
    plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
    print(f'Saved generated figure to {plot_filename}')

    plt.show()

if __name__ == '__main__':
    main()
