import numpy as np

from gp import *
from gp_ensemble import GPEnsemble
from DataLoaderGP import DataLoaderGP
import time
import casadi as cs

import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", type=int, required=False, default=1, help="Save the model? 1: yes, 0: no")
    args = parser.parse_args()
    

    training_dataset_filepath = '../data/training_dataset.pkl'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    source_simulator = 'gazebo_simulation'
    
    training_dataset_filepath = os.path.join(dir_path, '../..', 'outputs', source_simulator, 'data/training_v20_a10_gp0.pkl')
    model_save_filepath = os.path.join(dir_path, '../..', 'outputs', source_simulator, 'gp_models/')

    n_training_samples = 20

    data_loader_gp = DataLoaderGP(training_dataset_filepath, number_of_training_samples=n_training_samples)

    z = data_loader_gp.X
    y = data_loader_gp.y


    z_train = data_loader_gp.X_train
    y_train = data_loader_gp.y_train

    #print(f'z_train.shape: {z_train.shape}')
    #print(f'y_train.shape: {y_train.shape}')

    ensemble_components = 3 
    gpe = GPEnsemble(ensemble_components)

    theta0 = [1,1,1] # Kernel variables

    #RBF = KernelFunction(np.eye(theta0[0]), theta0[1])

    for n in range(ensemble_components):
        
        gpr = GPR(z_train[:,n], y_train[:,n], covariance_function=KernelFunction, theta=theta0)
        gpe.add_gp(gpr, n)


    print(gpe)
    gpe.fit()
    print(gpe)

    gpe.plot_gpe(z_train, y_train)

    if args.save==1:
        gpe.save(model_save_filepath)

    #gpe_loaded = GPEnsemble(3)
    #print(model_loaded.theta)
    #gpe_loaded.load(model_save_filepath)

    #print(gpe_loaded)






    #plt.savefig('../img/gpe_interpolation.pdf', format='pdf')
    #plt.savefig('../docs/gpe_interpolation.jpg', format='jpg')




if __name__ == "__main__":
    main()