import numpy as np
import matplotlib.pyplot as plt
from gp import *
from gp_ensemble import GPEnsemble
from DataLoaderGP import DataLoaderGP
import time
import casadi as cs
import seaborn as sns
import os


def main():

    training_dataset_filepath = '../data/training_dataset.pkl'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    source_simulator = 'gazebo_simulation'
    
    training_dataset_filepath = os.path.join(dir_path, '../..', 'outputs', source_simulator, 'data/static_dataset.pkl')
    model_save_filepath = os.path.join(dir_path, '../..', 'outputs', source_simulator, 'gp_models/')

    n_training_samples = 20

    data_loader_gp = DataLoaderGP(training_dataset_filepath, number_of_training_samples=n_training_samples)

    z = data_loader_gp.X
    y = data_loader_gp.y


    z_train = data_loader_gp.X_train
    y_train = data_loader_gp.y_train

    print(f'z_train.shape: {z_train.shape}')
    print(f'y_train.shape: {y_train.shape}')

    ensemble_components = 3 
    gpe = GPEnsemble(ensemble_components)

    theta0 = [1,1,1] # Kernel variables

    #RBF = KernelFunction(np.eye(theta0[0]), theta0[1])

    for n in range(ensemble_components):
        
        gpr = GPR(z_train[:,n], y_train[:,n], covariance_function=KernelFunction, theta=theta0)
        gpe.add_gp(gpr, n)



    gpe.fit()
    y_pred = gpe.predict(z_train)

    z_query = np.concatenate([np.arange(-30,30,0.5).reshape(-1,1) for i in range(3)], axis=1)
    y_query, std_query = gpe.predict(z_query, std=True)


    

    gpe.save(model_save_filepath)

    gpe_loaded = GPEnsemble(3)
    #print(model_loaded.theta)
    gpe_loaded.load(model_save_filepath)

    print(gpe_loaded)



    xyz = ['x','y','z']
    #plt.style.use('seaborn')
    sns.set_theme()
    plt.figure(figsize=(10, 6), dpi=100)

    for col in range(y_pred.shape[1]):
        #print(np.ravel([f_grads[col](z_query[:,col])[d,d].full() for d in range(z_query.shape[0])]))
        plt.subplot(1,3,col+1)
        plt.plot(z_query[:,col], y_query[:,col])
        plt.scatter(z_train[:,col], y_pred[:,col], marker='+', c='g')
        plt.xlabel(f'Velocity {xyz[col]} [ms-1]')
        plt.ylabel(f'Drag acceleration {xyz[col]} [ms-2]')
        plt.legend(('m(z) interpolation', "m(z') training"))
        plt.fill_between(z_query[:,col], y_query[:,col] - 2*std_query[col], y_query[:,col] + 2*std_query[col], color='gray', alpha=0.2)
    plt.tight_layout()
    plt.show()
    #plt.savefig('../img/gpe_interpolation.pdf', format='pdf')
    #plt.savefig('../docs/gpe_interpolation.jpg', format='jpg')




if __name__ == "__main__":
    main()