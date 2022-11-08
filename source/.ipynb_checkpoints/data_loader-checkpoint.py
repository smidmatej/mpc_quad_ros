from numpy.random import default_rng
from six.moves import cPickle as pickle #for performance
rng = default_rng()

import numpy as np

from sklearn.mixture import GaussianMixture
import scipy.stats


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


class data_loader:
    def __init__(self, filename, compute_reduction, number_of_training_samples):
        
        self.dictionary = load_dict(filename)
        
        # takes every *compute_reduction* index along first axis 
        self.compute_reduction = compute_reduction
        self.number_of_training_samples = number_of_training_samples
        
        self.p = self.dictionary['p'][1::compute_reduction,:]
        self.q = self.dictionary['q'][1::compute_reduction,:]
        self.v = self.dictionary['v'][1::compute_reduction,:]
        self.w = self.dictionary['w'][1::compute_reduction,:]
        self.u = self.dictionary['u'][1::compute_reduction,:]
        
        self.a_validation = self.dictionary['aero_drag'][1::compute_reduction]
        self.v_pred = self.dictionary['v_pred'][1::compute_reduction]
        
        assert self.p.shape[0] > number_of_training_samples , f"Not enough samples for requested number of training samples, try reducting the compute_reduction"

        # error in acceleration between measured and predicted is the regressed variable we are trying to estimate
        self.calculate_errors()
        
        self.z = np.concatenate((self.p, self.q, self.v, self.w, self.u),axis=1)
        
        #self.data = np.concatenate((self.z, self.y, self.a_validation), axis=1) # concatenates along the state axis
        
        
        # subsample further into *number_of_training_samples* samples for training
        #self._n_sub = int(self.z.shape[0]/number_of_training_samples)
        
        #self.shuffle()
        #print(self.z.shape)
        self.cluster_data()
        
        
    def shuffle(self):
        """
        Shuffles the data so that we can take a equidistant subsampling without loss of information
        Redundant after adding cluster_data()
        """
        # all relevant indexes
        idx = np.arange(self.z.shape[0])
        # random permutation of indexes
        np.random.shuffle(idx)
        
        # shuffle samples using the permuted indexes
        self.z = self.z[idx,:]
        self.y = self.y[idx,:]
        self.a_validation = self.a_validation[idx,:]

        
    def calculate_errors(self):
        """
        Calculates the first differential between measured and predicted velocity for the next time step.
        The calculated value is set as self.y
        """
        self.y = (self.v - self.v_pred)/self.dictionary['dt']
        self.y_x, self.y_y, self.y_z = np.split(self.y, 3, axis = 1) # divides into xyz dimensions
        

    def get_z(self, training=False):
        if training:
            #return self.data[::self._n_sub, :self.z.shape[1]]
            try:
                return self.representatives['z']
            except NameError: print('Need to cluster the data first')
        else:
            #return self.data[:,:self.z.shape[1]]
            return self.z
        
    def get_y(self, training=False):
        if training:
            try:
                return self.representatives['y']
            except NameError: print('Need to cluster the data first')
            #return self.data[::self._n_sub, self.z.shape[1]:self.z.shape[1]+self.y.shape[1]]
        else:
            #return self.data[:, self.z.shape[1]:self.z.shape[1]+self.y.shape[1]]
            return self.y
        
    def get_a_validation(self):
        #return self.data[:, self.z.shape[1]+self.y.shape[1]:]
        return self.a_validation
    
    
    
    def cluster_data(self):
        """
        Fits a Gaussian Miture Model to the state data. Sets the representatives property that gets used when getting training data. 
        The representatives are the samples with the highest probability of falling into their respective cluster.
        Clustering is done over the whole state space -- Care needed when using only a subspace
        """
        self.GMM = GaussianMixture(n_components=self.number_of_training_samples, random_state=0, n_init=3, init_params='kmeans').fit(self.z)
        
        # chooses the most representative samples to use as training samples
        self.representatives = dict()
        self.representatives['z'] = np.empty(shape=(self.GMM.n_components, self.z.shape[1]))
        self.representatives['y'] = np.empty(shape=(self.GMM.n_components, self.y.shape[1]))
        for i in range(self.GMM.n_components):
            # PDF of each sample
            density = scipy.stats.multivariate_normal(cov=self.GMM.covariances_[i], mean=self.GMM.means_[i]).logpdf(self.z)
            # Index of the sample with the max of PDF
            idx_most_rep = np.argmax(density)
            # Used as training data
            self.representatives['z'][i, :] = self.z[idx_most_rep,:]
            self.representatives['y'][i, :] = self.y[idx_most_rep,:]

    


