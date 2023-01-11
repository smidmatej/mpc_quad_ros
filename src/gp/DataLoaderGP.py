import numpy as np

from sklearn.mixture import GaussianMixture
import scipy.stats

from warnings import warn
import sys

# Adds the parent directory to the path so that we can import utils
sys.path.append('../')
from utils.save_dataset import load_dict
from utils.utils import v_dot_q, quaternion_inverse

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


class DataLoaderGP:
    def __init__(self, filepath : str, number_of_training_samples : int = 10):
        """
        Loading and preprocessing for the GP.
        """

        
        self.number_of_training_samples = number_of_training_samples
        self.filepath = filepath

        # ---------------- Open filepath and save its contents internally ---------------- 
        self.data_dict = load_dict(self.filepath)
        print(f'Loaded data from {self.filepath}')
        print(f'Number of samples: {self.data_dict["x_odom"].shape[0]}')

        print(f'Number of collocation points: {self.number_of_training_samples}')
        # ---------------- Make the appropriate transformations to extract X, y  ---------------- 
        self.X, self.y = self.preprocess_data()
        # ---------------- Select most informative samples ---------------- 
        self.X_train, self.y_train = self.cluster_data3D(self.X, self.y)

    """
    def load_data(self):
        '''
        Loads the dictonary at self.filepath into self.data_dict
        '''
        self.data_dict = load_dict(self.filepath)
        print(f'Loaded data from {self.filepath}')
        print(f'Number of samples: {self.data_dict["x_odom"].shape[0]}')
    """

    def preprocess_data(self):
        """
        Selects the v_body as a X feature vector and the acceleration error as y
        :returns: X, y 
        """
        v_world = self.data_dict['x_odom'][:,7:10]
        v_world_pred = self.data_dict['x_pred_odom'][:,7:10]

        q = self.data_dict['x_odom'][:,3:7]
        q_pred = self.data_dict['x_pred_odom'][:,3:7]

        # ---------------- Transform to body frame ----------------
        self.v_body = np.empty(v_world.shape)
        self.v_body_pred = np.empty(v_world_pred.shape)
        for i in range(v_world.shape[0]):
            self.v_body[i,:] = v_dot_q(v_world[i,:].reshape((-1,)), quaternion_inverse(q[i,:].reshape((-1,))))
            self.v_body_pred[i,:] = v_dot_q(v_world_pred[i,:].reshape((-1,)), quaternion_inverse(q_pred[i,:].reshape((-1,))))


        # First differences of t to get the dt between samples. Odom is sampled on average with 100Hz.
        dt = np.diff(self.data_dict['t_odom'])


        # ---------------- Calculate the acceleration error ----------------
        self.y = np.empty((self.v_body.shape[0]-1, 3))
        for dim in range(3):
            # dt is one sample shorter than the other data
            self.y[:,dim] = (self.v_body[1:,dim] - self.v_body_pred[:-1,dim]) / dt
        
        self.X = self.v_body[:-1,:] # input is the measured velocity

        return self.X, self.y


        


    def plot(self, filepath=None, show=False):
        xyz = ['x','y','z']
        #plt.style.use('seaborn')
        sns.set_theme()
        plt.figure(figsize=(10, 6), dpi=100)

        for col in range(self.v_body.shape[1]):
            #print(np.ravel([f_grads[col](z_query[:,col])[d,d].full() for d in range(z_query.shape[0])]))
            plt.subplot(1,3,col+1)
            plt.scatter(self.X[:,col], self.y[:,col], s=0.1, label='Samples')
            plt.scatter(self.X_train[:,col], self.y_train[:,col], marker='+', c='k', label='Collocation points')    
            #plt.scatter(self.X_train[:,col], self.y_train[:,col], s=100, marker='o', c='k')
            plt.xlabel(f'Velocity {xyz[col]} [ms-1]')
            plt.ylabel(f'Drag acceleration {xyz[col]} [ms-2]')
            plt.legend()
            #plt.legend(('m(z) interpolation', "m(z') training"))

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, format="pdf", bbox_inches="tight")
        if show:
            plt.show()

    def cluster_data1D(self, X, y):
        """
        Fits a Gaussian Miture Model to dimension dz and returns the samples (z[,dz], y[,dy]) that have the highest probability in the mixture model
        """
        GMM = GaussianMixture(n_components=self.number_of_training_samples, random_state=0, n_init=3, init_params='kmeans')
        GMM.fit(X)
        
        # chooses the most representative samples to use as training samples
        representatives = {'X': list(), 'y': list()}

        for i in range(GMM.n_components):
            # PDF of each sample
            density = scipy.stats.multivariate_normal(cov=GMM.covariances_[i], mean=GMM.means_[i]).logpdf(X)
            # Index of the sample with the max of PDF
            idx_most_rep = np.argmax(density)
            # Used as training data
            representatives['X'].append(X[idx_most_rep])
            representatives['y'].append(y[idx_most_rep])
        return representatives
    
    def cluster_data3D(self, X, y):
        """
        Fits a Gaussian Miture Model to each dimension of dz and returns the samples (z[,dz], y[,dy]) that have the highest probability in the mixture model
        """
        representatives = {'X': list(), 'y': list()}
        for i in range(3):
            representatives1D = self.cluster_data1D(X[:,i].reshape((-1,1)), y[:,i].reshape((-1,1)))
            representatives['X'].append(representatives1D['X'])
            representatives['y'].append(representatives1D['y'])
        
        X_rep = np.squeeze(np.array(representatives['X']).T)
        y_rep = np.squeeze(np.array(representatives['y']).T)
        return X_rep, y_rep

