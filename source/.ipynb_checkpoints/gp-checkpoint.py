import numpy as np
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

class KernelFunction:
    def __init__(self,L=np.eye(10),sigma_f=1):
        """
        Contructor of the RBF function k(x1,x2).
        
        :param: L: Square np.array of dimension d x d. Defines the length scale of the kernel function
        :param: sigma_f: Scalar value used to lineary scale the amplidude of the k(x,x)
        """
        self.L = L
        self.sigma_f = sigma_f
        
    def __call__(self, x1, x2):
        """
        Calculate the value of the kernel function given 2 input vectors
        
        :param: x1: np.array of dimension 1 x d
        :param: x2: np.array of dimension 1 x d
        """
        difference = x1-x2
        
        return float(self.sigma_f**2 * np.exp(-1/2*difference.T.dot(np.linalg.inv(self.L*self.L)).dot(difference)))

    def __str__(self):
        return f"L = {self.L}, \n\r Sigma_f = {self.sigma_f}"


    
    
class GPR:
    def __init__(self, z_train, y_train, covariance_function, theta):
        """
        The whole contructor is in this method. This method is called when contructing this class and after self.fit()
        
        :param: z_train: np.array of n samples with dimension d. Input samples for regression
        :param: y_train: np.array of n samples with dimension d. Output samples for regression
        :param: covariance_function: Reference to a KernelFunction
        :param: theta: np.array of hyperparameters
        """

        self.initialize(z_train, y_train, covariance_function, theta)

        
    def initialize(self, z_train, y_train, covariance_function, theta):
        """
        The whole contructor is in this method. This method is called when contructing this class and after self.fit()
        
        :param: z_train: np.array of n samples with dimension d. Input samples for regression
        :param: y_train: np.array of n samples with dimension d. Output samples for regression
        :param: covariance_function: Reference to a KernelFunction
        :param: theta: np.array of hyperparameters
        """
        
        if z_train is None or y_train is None:
            self.n_train = 0
            self.z_dim = 1 # this needs to be set in a general way for prediction from prior
        else:
            self.n_train = z_train.shape[0]
            self.z_dim = z_train.shape[1]
            
        self.z_train = z_train
        self.y_train = y_train
        self.covariance_function = covariance_function
        
        self.theta=theta
        
        self.kernel = covariance_function(L=np.eye(self.z_dim)*theta[0], sigma_f=theta[-2])
        
        self.noise = theta[-1]
        
        self.inv_cov_matrix_of_input_data = np.linalg.inv(
            self.calculate_covariance_matrix(z_train, z_train, self.kernel) \
            + (self.noise+1e-7)*np.identity(self.n_train))
        
        print(f'Size of feature training data = {(self.n_train, self.z_dim)}')
        print(f'Size of output training data = {self.n_train, self.z_dim}')


    def predict(self, at_values_z, var=False, std=False, cov=False):
        """
        Evaluate the posterior mean m(z) and covariance sigma for supplied values of z
        
        :param at_values_z: np vector to evaluate at
        """
        sigma_k = self.calculate_covariance_matrix(self.z_train, at_values_z, self.kernel)
        sigma_kk = self.calculate_covariance_matrix(at_values_z, at_values_z, self.kernel)
        #print(sigma_kk)

        if self.n_train == 0:
            mean_at_values = np.zeros((at_values_z.shape[0],1))
        
            cov_matrix = sigma_kk
        
        else:
            mean_at_values = sigma_k.T.dot(
                                    self.inv_cov_matrix_of_input_data.dot(
                                        self.y_train)) 

            cov_matrix = sigma_kk - sigma_k.T.dot(
                                    self.inv_cov_matrix_of_input_data.dot(
                                        sigma_k))


        variance = np.diag(cov_matrix)
        self._memory = {'mean': mean_at_values, 'covariance_matrix': cov_matrix, 'variance':variance}
        if cov:
            return mean_at_values, cov_matrix
        elif var:
            return mean_at_values, variance
        elif std:
            return mean_at_values, np.sqrt(variance)
        else:  
            return mean_at_values
        
    #def predict_symbolic(self, at_values_z_sym):
        
    def draw_function_sample(self, at_values_z, n_sample_functions=1):
        """ 
        Draw a function from the current distribution evaluated at at_values_z.
        
        :param at_values_z: np vector to evaluate function at
        :param n_sample_functions: allows for multiple sample draws from the same distribution
        """

        mean_at_values, cov_matrix = self.predict(at_values_z, cov=True)
        y_sample = np.random.multivariate_normal(mean=mean_at_values.ravel(), cov=cov_matrix, size=n_sample_functions)
        return y_sample
        
    def fit(self):
        """ 
        Uses the negative log likelyhood function to maximize likelyhood by varying the hyperparameters theta
        """
        
        low_bnd = 0.01
        #bnds = tuple([(low_bnd, None) for i in range(self.z_train.shape[1])]) + ((low_bnd, None), (low_bnd, None))
        bnds = ((low_bnd, None), (low_bnd, None), (low_bnd, None))
        print('Maximizing the likelyhood function for GP')
        print(f'Hyperparameters before optimization = {self.theta}')
        
        sol_min = minimize(self.nll, x0=self.theta, method='L-BFGS-B', bounds=bnds)
        theta_star = sol_min.x
        
        # Ammounts to recreating all relevant contents of this class
        self.initialize(self.z_train, self.y_train, self.covariance_function, theta_star)

        print('Optimization done')
        print(f'Hyperparameters after optimization = {self.theta}')
        
    def nll(self, theta):
        """
        Numerically more stable implementation of Eq. (11)
        as described in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section 2.2, Algorithm 2.1.
        
        :param: theta: Hyperparameter np.array
        """

        # Kernel function k(x1,x2)
        k = self.covariance_function(L=np.eye(self.z_train.shape[1])*theta[0], sigma_f=theta[-2])
        
        # Evaluate k(x,x) over all combinations of x1 and x2
        K = self.calculate_covariance_matrix(self.z_train, self.z_train, k) + \
                (theta[-1]+1e-7)*np.identity(self.z_train.shape[0])
        
        L = cholesky(K)

        S1 = solve_triangular(L, self.y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)


        neg_log_lklhd = (np.sum(np.log(np.diagonal(L))) + \
                       0.5 * self.y_train.T.dot(S2) + \
                       0.5 * self.z_train.shape[0] * np.log(2*np.pi)).flatten()
        return neg_log_lklhd
        
    @staticmethod
    def calculate_covariance_matrix(x1,x2, kernel):
        """
        Fills in a matrix with k(x1[i,:], x2[j,:])
        
        :param: x1: n x d np.array, where n is the number of samples and d is the dimension of the regressor
        :param: x2: n x d np.array, where n is the number of samples and d is the dimension of the regressor
        :param: kernel: Instance of a KernelFunction class
        """
        
        if x1 is None or x2 is None:
            # Dimension zero matrix 
            return np.zeros((0,0))
        
        cov_mat = np.empty((x1.shape[0], x2.shape[0]))*np.NaN
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        
        # for all combinations calculate the kernel
        for i in range(x1.shape[0]):
            a = x1[i,:].reshape(-1,1)
            for j in range(x2.shape[0]):

                b = x2[j,:].reshape(-1,1)

                
                cov_mat[i,j] = kernel(a,b)

        return cov_mat


