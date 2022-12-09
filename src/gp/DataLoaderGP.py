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


class DataLoaderGP:
    def __init__(self, filename, number_of_training_samples=10):
        


        self.number_of_training_samples = number_of_training_samples
        self.data_dict = load_dict(filename)
        print(f'Loaded data from {filename}')

        # Data dict stores only the world frame velocities
        print(f"self.data_dict['x_odom'][:,7:10].shape {self.data_dict['x_odom'][:,7:10].shape}")
        print(f"self.data_dict['x_odom'][:,3:7].shape {self.data_dict['x_odom'][:,3:7].shape}")
        print(f"self.data_dict['x_pred_odom'][:,7:10].shape {self.data_dict['x_pred_odom'][:,7:10].shape}")
        print(f"self.data_dict['x_pred_odom'][:,3:7].shape {self.data_dict['x_pred_odom'][:,3:7].shape}")



        v_world = self.data_dict['x_odom'][:,7:10]
        v_world_pred = self.data_dict['x_pred_odom'][:,7:10]

        q = self.data_dict['x_odom'][:,3:7]
        q_pred = self.data_dict['x_pred_odom'][:,3:7]

        '''
                fig = plt.figure()
        for k in range(3):
            ax = fig.add_subplot(3,1,k+1)
            ax.plot(v_world[:,k], label='v_world')
            ax.plot(v_world_pred[:,k], label='v_world_pred')
            ax.legend()
        plt.title('v_world')
        plt.show()
        '''    

        

        '''
        fig = plt.figure()
        for k in range(4):
            ax = fig.add_subplot(4,1,k+1)
            ax.plot(q[:,k], label='q')
            ax.plot(q_pred[:,k], label='q_pred')
            ax.legend()
        plt.title('q')
        plt.show()
        '''

        
        
        self.v_body = np.empty(v_world.shape)
        self.v_body_pred = np.empty(v_world_pred.shape)
        for i in range(v_world.shape[0]):
            self.v_body[i,:] = v_dot_q(v_world[i,:].reshape((-1,)), quaternion_inverse(q[i,:].reshape((-1,))))
            self.v_body_pred[i,:] = v_dot_q(v_world_pred[i,:].reshape((-1,)), quaternion_inverse(q_pred[i,:].reshape((-1,))))

        '''
        fig = plt.figure()
        for k in range(3):
            ax = fig.add_subplot(3,1,k+1)
            ax.plot(self.v_body[:,k])
            ax.plot(self.v_body_pred[:,k])
        plt.title('v_body')
        plt.show()
        '''

        

        #self.v_body = self.data_dict['x_odom'][:,7:10]
        #self.v_body_pred = self.data_dict['x_pred_odom'][:,7:10]
        # First differences of t to get the dt between samples. Odom is sampled on average with 100Hz.
        dt = np.diff(self.data_dict['t_odom'])
        #dt = 0.01
        '''
        fig = plt.figure()
        plt.plot(dt)
        plt.title('dt')
        plt.show()
        '''

        
       
        
        print(f'v_body.shape {self.v_body.shape}')
        print(f'v_body_pred.shape {self.v_body_pred.shape}')
        # dt is one sample shorter than the other data
        self.y = np.empty((self.v_body.shape[0]-1, 3))
        for dim in range(3):
            self.y[:,dim] = (self.v_body[1:,dim] - self.v_body_pred[:-1,dim]) / dt
        
        '''
        fig = plt.figure()
        for k in range(3):
            ax = fig.add_subplot(3,1,k+1)
            ax.plot(self.y[:,k], label='y')
            ax.plot(self.v_body[:,k], '--', label='v_body')
            #ax.plot(self.data_dict['aero_drag'][:,k])
        plt.title('acc')
        plt.legend()
        plt.show()
        '''

        
        
        fig = plt.figure('a_error(v_body)')
        for k in range(3):
            ax = fig.add_subplot(3,1,k+1)
            ax.scatter(self.v_body[:-1,k], self.y[:,k], s=0.1, label='y')
            #ax.plot(self.data_dict['aero_drag'][:,k])
        plt.show()
        

        #self.y = np.array([(self.v_body[:-1,dim] - self.v_body_pred[:-1,dim])/dt for dim in range(3)]).T # error in velocity between measured and predicted is the regressed variable we are trying to estimate
        print(f'self.y {self.y.shape}')
        self.X = self.v_body[:-1,:] # input is the measured velocity

        # USING STANDARD X,y notations     
        self.X_train, self.y_train = self.cluster_data3D(self.X, self.y)
        '''
        plt.plot(self.data_dict['x_odom'][:,0],'--')
        plt.plot(self.v_body[:,0])
        plt.plot(self.v_body_pred[:,0])
        plt.plot(self.y[:,0],'+-')
        plt.show()
        '''
        '''
        print(f'self.X_train {self.X_train.shape}')
        plt.plot(self.X_train[:,0], self.y_train[:,0], 'o')
        plt.plot(self.X[:,0], self.y[:,0])
        plt.show()
        '''

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

