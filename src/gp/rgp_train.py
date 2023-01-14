 # 
 # This file is part of the mpc_quad_ros distribution (https://github.com/smidmatej/mpc_quad_ros).
 # Copyright (c) 2023 Smid Matej.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #
 
import numpy as np
import time
import casadi as cs

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from RGP import RGP
    from GPE import GPEnsemble
    from DataLoaderGP import DataLoaderGP
except ImportError:
    from gp.RGP import RGP
    from gp.GPE import GPEnsemble
    from gp.DataLoaderGP import DataLoaderGP

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", type=int, required=False, default=1, help="Save the model? 1: yes, 0: no")
    args = parser.parse_args()

    #environment = 'python_simulation'
    environment = 'gazebo_simulation'
    
    filename = 'training_v20_a10_gp0'
    #filename = 'training_dataset'
    training_dataset_filepath = os.path.join(dir_path, '../..', 'outputs', environment, 'data', filename + '.pkl')
    model_save_filepath = os.path.join(dir_path, '../..', 'outputs', environment, 'gp_models/')

    if args.save != 1:
        model_save_filepath = None
    gpefit_plot_filepath = os.path.join(dir_path, '../..', 'outputs', 'graphics', 'rgpefit_' + filename + '.pdf')
    gpesamples_plot_filepath = os.path.join(dir_path, '../..', 'outputs', 'graphics', 'rgpesamples_' + filename + '.pdf')

    n_training_samples = 20
    theta0 = [1.0,1.0,1]*3 # Kernel variables

    train_rgp(training_dataset_filepath, model_save_filepath, n_training_samples=n_training_samples, show_plots=True, gpefit_plot_filepath=gpefit_plot_filepath, gpesamples_plot_filepath=gpesamples_plot_filepath)


def train_rgp(training_dataset_filepath, model_save_filepath, n_training_samples=10, theta0=None, show_plots=True, gpefit_plot_filepath=None, gpesamples_plot_filepath=None):
    """
    Instantiates the DataLoaderGP to get data from a file and obtain training samples. 
    Then load the training samples into GPE and do hyperparameter optimization on them.
    """
    ensemble_components = 3 

    data_loader_gp = DataLoaderGP(training_dataset_filepath, number_of_training_samples=n_training_samples)

    

    if theta0 is not None:
        assert len(theta0) == ensemble_components, f"theta0 has to be a list of len {ensemble_components}, passed a list of {len(theta0)}"

    #breakpoint()
    # -------------- Fill GPE with data from the appropriate dimension --------------
    X_ = np.arange(-10.0,10.0,1.0)
    y_ = np.random.normal(0, 0, size=X_.shape)
    gps = [None]*ensemble_components
    for n in range(ensemble_components):
        if theta0 is None:
            # I dont have a guess of the theta0 parameters -> Use the default in GP
            gps[n] = RGP(X_, y_)
        else:
            gps[n] = RGP(data_loader_gp.X_train[:,n].reshape(-1,1), data_loader_gp.y_train[:,n].reshape(-1,1), theta=theta0[n])
        
    gpe = GPEnsemble.fromlist(gps)


    # TODO: Implement regression. Perhaps encapsulate the for loop inside the RGP class

    print('Training model recursively...')
    pbar = tqdm(total=data_loader_gp.X.shape[0])
    
    for t in range(data_loader_gp.X.shape[0]):
        for n in range(ensemble_components):
            gpe.gp[n].regress(np.atleast_1d(data_loader_gp.X[t,n]), np.atleast_1d(data_loader_gp.y[t,n]))
        pbar.update()
    pbar.close()
    print('Done training model.')

    # -------------- Plotting --------------

    # Color scheme convert from [0,255] to [0,1]
    cs = [[x/256 for x in (8, 65, 92)], \
            [x/256 for x in (204, 41, 54)], \
            [x/256 for x in (118, 148, 159)], \
            [x/256 for x in (232, 197, 71)]] 



    plt.style.use('fast')
    sns.set_style("whitegrid")

    X_query = np.linspace(-10.0, 10.0, 100)

    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = [None]*3

    ax[0] = fig.add_subplot(131)
    ax[1] = fig.add_subplot(132)
    ax[2] = fig.add_subplot(133)

    for n in range(ensemble_components):
        mean, std = gpe.gp[n].predict(X_query, std=True)
        ax[n].plot(X_query, mean, '--', color=cs[0], label='E[g(x)]')
        ax[n].scatter(gpe.gp[n].X, gpe.gp[n].mu_g_t, marker='o', s=20, color=cs[0], label='Basis Vectors')
        ax[n].fill_between(X_query.ravel(), mean - 2*std, mean + 2*std, color=cs[0], alpha=0.2, label='2 std')

        
        ax[n].scatter(data_loader_gp.X[:,n], data_loader_gp.y[:,n], s=0.5, marker='.', color=cs[2], label='Samples')

        ax[n].set_xlabel('x')
        ax[n].set_ylabel('y')
        ax[n].set_title(f'GP {n+1}')
        ax[n].grid(True)
        
    
    plt.savefig('rpg_trained.png')
    plt.show()

    # -------------- Save GPE after hyperparameter optimization --------------
    if model_save_filepath is not None:
        gpe.save(model_save_filepath)
    
    # -------------- Run, save and show plots --------------
    if gpesamples_plot_filepath is not None:
        data_loader_gp.plot(gpesamples_plot_filepath, show=show_plots)
    if gpefit_plot_filepath is not None:
        gpe.plot(data_loader_gp.X_train, data_loader_gp.y_train, gpefit_plot_filepath, show=show_plots)
        





if __name__ == "__main__":
    main()