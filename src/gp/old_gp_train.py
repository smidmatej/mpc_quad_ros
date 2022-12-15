import numpy as np
import matplotlib.pyplot as plt
from gp import *
from gp_ensemble import GPEnsemble
from data_loader import data_loader
import time
import casadi as cs
import seaborn as sns
import os


def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    training_dataset_filepath = os.path.join(dir_path, '../..', 'outputs/python_simulation/data/simulated_trajectory.pkl')
    compute_reduction = 1
    n_training_samples = 20

    d_loader = data_loader(training_dataset_filepath, compute_reduction=compute_reduction, number_of_training_samples=n_training_samples, body_frame=True)               

    z = d_loader.get_z(training=False)
    y = d_loader.get_y(training=False)


    z_train = d_loader.get_z(training=True)
    y_train = d_loader.get_y(training=True)


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


    model_save_fname = "models/ensemble"

    gpe.save(model_save_fname)

    gpe_loaded = GPEnsemble(3)
    #print(model_loaded.theta)
    gpe_loaded.load(model_save_fname)

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