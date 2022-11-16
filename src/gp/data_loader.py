from numpy.random import default_rng
import pickle as pkl
rng = default_rng()

import numpy as np

from sklearn.mixture import GaussianMixture
import scipy.stats

from warnings import warn





def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pkl.load(f)
    return ret_di


class data_loader:
    def __init__(self, filename, compute_reduction=1, number_of_training_samples=10, sample_to_amount=False, amount_of_samples=0, body_frame=False):
        

        # takes every *compute_reduction* index along first axis 
        self.compute_reduction = compute_reduction
        self.number_of_training_samples = number_of_training_samples
        self.dictionary = load_dict(filename)
        if sample_to_amount:
            # compute compute_reduction so that we have amount_of_samples samples
            self.compute_reduction = int(self.dictionary['p'].shape[0]/amount_of_samples)
            print(f'compute_reduction = {self.compute_reduction}')
        else:
            self.compute_reduction = compute_reduction

        self.p = self.dictionary['p'][1::self.compute_reduction,:]
        self.q = self.dictionary['q'][1::self.compute_reduction,:]
        if body_frame:
            # For training GPs
            self.v = self.dictionary['v_body'][1::self.compute_reduction,:]
        else:
            # Not for training GPs
            self.v = self.dictionary['v'][1::self.compute_reduction,:]
        self.w = self.dictionary['w'][1::self.compute_reduction,:]
        self.u = self.dictionary['u'][1::self.compute_reduction,:]

        self.t = self.dictionary['t'][1::self.compute_reduction]
        
        self.a_validation = self.dictionary['aero_drag'][1::self.compute_reduction]
        self.v_pred = self.dictionary['v_pred'][1::self.compute_reduction]
        
        assert self.p.shape[0] > number_of_training_samples , f"Not enough samples for requested number of training samples, \
            self.p.shape = {self.p.shape}, number_of_training_samples = {number_of_training_samples}"
        # error in acceleration between measured and predicted is the regressed variable we are trying to estimate
        self.calculate_errors()
        
        self.z = np.concatenate((self.p, self.q, self.v, self.w, self.u),axis=1)
        

        self.representatives = self.cluster_data_dimensions_concatenate([7,8,9], [0,1,2])
        
        
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
        
        warn('This is deprecated, use dimension-wise clustering', DeprecationWarning, stacklevel=2)
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

    def cluster_data_dimension(self, dz, dy):
        """
        Fits a Gaussian Miture Model to dimension dz and returns the samples (z[,dz], y[,dy]) that have the highest probability in the mixture model
        """
        self.GMM = GaussianMixture(n_components=self.number_of_training_samples, random_state=0, n_init=3, init_params='kmeans').fit(self.z[:,dz].reshape(-1,1))
        
        # chooses the most representative samples to use as training samples
        representatives = dict()
        representatives = dict()
        representatives['z'] = np.empty(shape=(self.GMM.n_components, 1))
        representatives['y'] = np.empty(shape=(self.GMM.n_components, 1))
        for i in range(self.GMM.n_components):
            # PDF of each sample
            density = scipy.stats.multivariate_normal(cov=self.GMM.covariances_[i], mean=self.GMM.means_[i]).logpdf(self.z[:,dz])
            # Index of the sample with the max of PDF
            idx_most_rep = np.argmax(density)
            # Used as training data
            representatives['z'][i] = self.z[idx_most_rep, dz]
            representatives['y'][i] = self.y[idx_most_rep, dy]
        return representatives
    
    def cluster_data_dimensions_concatenate(self, dimensions_z, dimensions_y):
        """
        Return a dictionary of representative samples in a list of dimensions. 
        Each dimension is seperate, the represantativeness of each sample is measured in that dimension only
        """
        assert len(dimensions_z)==len(dimensions_y), f"dimensions_z = {dimensions_z}, dimensions_y = {dimensions_y}"
        reps = [None]*len(dimensions_z)
        reps_concat = dict()
        for index in range(len(dimensions_z)):
            reps[index] = self.cluster_data_dimension(dimensions_z[index], dimensions_y[index])

        reps_concat['z'] = np.concatenate([reps[i]['z'] for i in range(len(reps))], axis=1)
        reps_concat['y'] = np.concatenate([reps[i]['y'] for i in range(len(reps))], axis=1)
        return reps_concat


