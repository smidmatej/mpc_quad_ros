import numpy as np

from sklearn.mixture import GaussianMixture
import scipy.stats

from warnings import warn
import sys

# Adds the parent directory to the path so that we can import utils
sys.path.append('../')
from utils.save_dataset import load_dict
from utils.utils import v_dot_q, quaternion_inverse



class DataLoaderGP:
    def __init__(self, filename, number_of_training_samples=10):
        


        self.number_of_training_samples = number_of_training_samples
        self.data_dict = load_dict(filename)

        # Data dict stores only the world frame velocities
        print(f"self.data_dict['x_odom'][:,7:10].shape {self.data_dict['x_odom'][:,7:10].shape}")
        print(f"self.data_dict['x_odom'][:,3:7].shape {self.data_dict['x_odom'][:,3:7].shape}")
        print(f"self.data_dict['x_pred_odom'][:,7:10].shape {self.data_dict['x_pred_odom'][:,7:10].shape}")
        print(f"self.data_dict['x_pred_odom'][:,3:7].shape {self.data_dict['x_pred_odom'][:,3:7].shape}")

        self.v_body = np.empty((self.data_dict['t_odom'].shape[0], 3))*np.NaN
        self.v_body_pred = np.empty((self.data_dict['t_odom'].shape[0], 3))*np.NaN
        for i in range(len(self.data_dict['x_odom'])):
            self.v_body[i,:] = v_dot_q(self.data_dict['x_odom'][i,7:10].reshape((-1,)), quaternion_inverse(self.data_dict['x_odom'][i,3:7].reshape((-1,))))
            self.v_body_pred[i,:] = v_dot_q(self.data_dict['x_pred_odom'][i,7:10].reshape((-1,)), quaternion_inverse(self.data_dict['x_pred_odom'][i,3:7].reshape((-1,))))

        # First differences of t to get the dt between samples. Odom is sampled on average with 100Hz.
        dt = np.diff(self.data_dict['t_odom'])
        # dt is one sample shorter than the other data
        self.y = np.array([(self.v_body[:-1,dim] - self.v_body_pred[:-1,dim])/dt for dim in range(3)]).T # error in velocity between measured and predicted is the regressed variable we are trying to estimate
        print(f'self.y {self.y.shape}')
        self.X = self.v_body[:-1,:] # input is the measured velocity

        # USING STANDART X,y notations     
        self.X_train, self.y_train = self.cluster_data3D(self.X, self.y)
        


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

