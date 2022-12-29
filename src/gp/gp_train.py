import numpy as np

from gp import *
from gp_ensemble import GPEnsemble
from DataLoaderGP import DataLoaderGP
import time
import casadi as cs

import os
import argparse


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", type=int, required=False, default=1, help="Save the model? 1: yes, 0: no")
    args = parser.parse_args()

    environment = 'gazebo_simulation'
    
    filename = 'test_circle_v15_a5_gp0'
    training_dataset_filepath = os.path.join(dir_path, '../..', 'outputs', environment, 'data', filename + '.pkl')
    model_save_filepath = os.path.join(dir_path, '../..', 'outputs', environment, 'gp_models/')

    if args.save != 1:
        model_save_filepath = None
    gpefit_plot_filepath = os.path.join(dir_path, '../..', 'outputs', 'graphics', 'gpefit_' + filename + '.pdf')
    gpesamples_plot_filepath = os.path.join(dir_path, '../..', 'outputs', 'graphics', 'gpesamples_' + filename + '.pdf')

    n_training_samples = 20
    theta0 = [1,1,1]*3 # Kernel variables

    train_gp(training_dataset_filepath, model_save_filepath, n_training_samples=n_training_samples, show_plots=True, save_plot=True, gpefit_plot_filepath=gpefit_plot_filepath, gpesamples_plot_filepath=gpesamples_plot_filepath)


def train_gp(training_dataset_filepath, model_save_filepath, n_training_samples=10, theta0=None, show_plots=True, gpefit_plot_filepath=None, gpesamples_plot_filepath=None):
    """
    Instantiates the DataLoaderGP to get data from a file and obtain training samples. 
    Then load the training samples into GPE and do hyperparameter optimization on them.
    """
    ensemble_components = 3 

    data_loader_gp = DataLoaderGP(training_dataset_filepath, number_of_training_samples=n_training_samples)

    gpe = GPEnsemble(ensemble_components)

    if theta0 is not None:
        assert len(theta0) == ensemble_components, f"theta0 has to be a list of len {ensemble_components}, passed a list of {len(theta0)}"


    # -------------- Fill GPE with data from the appropriate dimension --------------
    for n in range(ensemble_components):
        if theta0 is None:
            # I dont have a guess of the theta0 parameters -> Use the default in GPR
            gpr = GPR(data_loader_gp.X_train[:,n], data_loader_gp.y_train[:,n], covariance_function=KernelFunction)
        else:
            gpr = GPR(data_loader_gp.X_train[:,n], data_loader_gp.y_train[:,n], covariance_function=KernelFunction, theta=theta0[n])
        
        gpe.add_gp(gpr, n)


    # -------------- Hyperparameter optimization --------------
    print(gpe)
    gpe.fit()
    print(gpe)

    # -------------- Save GPE after hyperparameter optimization --------------
    if model_save_filepath is not None:
        gpe.save(model_save_filepath)
    
    # -------------- Run, save and show plots --------------
    if gpesamples_plot_filepath is not None:
        data_loader_gp.plot(gpesamples_plot_filepath, show=show_plots)
    if gpefit_plot_filepath is not None:
        gpe.plot(data_loader_gp.X_train[:,n], data_loader_gp.y_train, gpefit_plot_filepath, show=show_plots)
        





if __name__ == "__main__":
    main()