import numpy as np
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import casadi as cs
import joblib

class KernelFunction:
    def __init__(self,L=np.eye(10),sigma_f=1):
        """
        Contructor of the RBF function k(x1,x2).
        
        :param: L: Square np.array of dimension d x d. Defines the length scale of the kernel function
        :param: sigma_f: Scalar value used to lineary scale the amplidude of the k(x,x)
        """
        self.L = L
        #self.L_inv = 

        #self.inv_cov_matrix_of_input_data = cs.solve(self.cov_matrix_of_input_data, cs.MX.eye(self.cov_matrix_of_input_data.size1()))
        self.sigma_f = sigma_f
        self.kernel_type = "SEK" # Squared exponential kernel
        self.params = {
                        "L": self.L,
                        "sigma_f": self.sigma_f
                        }
        
    def __call__(self, x1, x2):
        """
        Calculate the value of the kernel function given 2 input vectors
        
        :param: x1: np.array or cs.MX of dimension 1 x d 
        :param: x2: np.array or cs.MX of dimension 1 x d 
        """
        assert self.kernel_type == "SEK", f"Kernel type is not of type SEK, kernel_type={self.kernel_type}"

        if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray): 
            # assure that x is not onedimensional
            x1 = np.atleast_2d(x1)
            x2 = np.atleast_2d(x2)

            dif = x1-x2
            return float(self.sigma_f**2 * np.exp(-1/2*dif.T.dot(np.linalg.inv(self.L*self.L)).dot(dif)))
        else:
            # input is assumed to be a casadi vector
            # Only implemented for scalar symbolics
            #print(self.L.shape)
            #assert self.L.shape == (1,1), "Symbolic kernel evaluation only works for n x 1 inputs, create a kernel with L.shape = (1,1)"
            assert x1.shape[1] == self.L.shape[0] and x2.shape[1] == self.L.shape[0] , f"Cannot multiply L with x1 or x2, L.shape={self.L.shape}, x1.shape={x1.shape}, x2.shape={x2.shape    }"

            assert x1.shape[1] == x2.shape[1], "Inputs to kernel need identical second axis dimensions"
            assert x1.shape[0] == 1 and x2.shape[0] == 1, f"Kernel function defined only for 1 x d casadi symbolics, x1.shape={x1.shape}, x2.shape={x2.shape    }"

            dif = x1-x2
            return self.sigma_f**2 * np.exp(-1/2* cs.mtimes(cs.mtimes(dif, np.linalg.inv(self.L*self.L)), dif.T))


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
        if z_train is None and y_train is None and covariance_function is None and theta is None:
            # do nothing
            1+1
        else:
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
            # No training data given, gp will only provide prior predictions
            self.n_train = 0
            self.z_dim = 1 # this needs to be set in a general way for prediction from prior
        else:
            # Assure that the training data has the correct format
            if z_train.ndim < 2 and z_train.shape != (1,1):
                # input is a (n,) np.array
                z_train = z_train.reshape((-1,1))
            else:
                # input is a scalar or ndim >=2
                y_train = np.atleast_2d(y_train)


            if y_train.ndim == 1 and y_train.shape != (1,1):
                # input is a (n,)
                y_train = y_train.reshape((-1,1))
            else:
                # input is a scalar or ndim >=2
                y_train = np.atleast_2d(y_train)


            self.n_train = z_train.shape[0]
            self.z_dim = z_train.shape[1]


        self.z_train = z_train
        self.y_train = y_train
        self.covariance_function = covariance_function
        
        self.theta=theta
        
        self.kernel = covariance_function(L=np.eye(self.z_dim)*theta[0], sigma_f=theta[-2])
        
        self.noise = theta[-1]
        
        if isinstance(z_train, cs.MX):
            self.cov_matrix_of_input_data = self.calculate_covariance_matrix(z_train, z_train, self.kernel) \
                                            + (self.noise+1e-7)*np.identity(self.n_train)

            # Symbolic matrix inverse using linear system solve, since cs does not have native matrix inverse method
            self.inv_cov_matrix_of_input_data = cs.solve(self.cov_matrix_of_input_data, cs.MX.eye(self.cov_matrix_of_input_data.size1()))

        else:
            self.inv_cov_matrix_of_input_data = np.linalg.inv(
                self.calculate_covariance_matrix(z_train, z_train, self.kernel) \
                + (self.noise+1e-7)*np.identity(self.n_train))
        
        #print(f'Size of feature training data = {(self.n_train, self.z_dim)}')
        #print(f'Size of output training data = {self.n_train, self.z_dim}')


    def predict(self, at_values_z, var=False, std=False, cov=False):
        """
        Evaluate the posterior mean m(z) and covariance sigma for supplied values of z
        
        :param at_values_z: np vector to evaluate at
        """

        ### TODO: Add std and variance to casadi prediction ###
        sigma_k = self.calculate_covariance_matrix(self.z_train, at_values_z, self.kernel)
        sigma_kk = self.calculate_covariance_matrix(at_values_z, at_values_z, self.kernel)


        if self.n_train == 0:
            mean_at_values = np.zeros((at_values_z.shape[0],1))
        
            cov_matrix = sigma_kk
        
        else:
            if isinstance(at_values_z, cs.MX):
                mean_at_values = cs.mtimes(sigma_k.T, cs.mtimes(self.inv_cov_matrix_of_input_data, self.y_train))
                cov_matrix = np.eye(1)

               
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
        
        print(self.y_train.shape)
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

    def jacobian(self, z):
        """
        Casadi symbolic jacobian of prediction y with respect to z

        :param: z: Casadi symbolic vector expression n x 1
        :return: Casadi function jacobian
        """
        assert z.shape[1] == 1, f"z needs to be n x 1, z.shape={z.shape}"
        y = self.predict(z)

        J = cs.jacobian(y, z)
        Jf = cs.Function('J', [z], [J])
        return Jf
        
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


    def save(self, path):
        """
        Saves the current GP  to the specified path as a pickle file. Must be re-loaded with the function load
        :param path: absolute path to save the regressor to
        """

        saved_vars = {
            "kernel_params": self.kernel.params,
            "kernel_type": self.kernel.kernel_type,
            "z_train": self.z_train,
            "y_train": self.y_train,
            "theta": self.theta,
            "z_dim": self.z_dim,
        }

        with open(path, 'wb') as f:
            joblib.dump(saved_vars, f)
        
    def load(self, path):
        """
        Load a pre-trained GP regressor
        :param data_dict: a dictionary with all the pre-trained matrices of the GP regressor
        """
        data_dict = joblib.load(path)

        #self.kernel.params = data_dict['kernel_params']
        #self.kernel.kernel_type = data_dict['kernel_type']
        '''
        self.z_train = data_dict['z_train']
        self.y_train = data_dict['y_train']
        self.theta = data_dict['theta']
        self.kernel = KernelFunction(data_dict['kernel_params']['L'], data_dict['kernel_params']['sigma_f'])
        self.sigma_n = self.theta[-1]
        '''
        self.initialize(data_dict['z_train'], data_dict['y_train'], KernelFunction, data_dict['theta'])
        

    def __str__(self):
        return f"Theta: {self.theta}"




    @staticmethod
    def calculate_covariance_matrix(x1,x2, kernel):
        """
        Fills in a matrix with k(x1[i,:], x2[j,:])
        
        :param: x1: n x d np.array, where n is the number of samples and d is the dimension of the regressor
        :param: x2: n x d np.array, where n is the number of samples and d is the dimension of the regressor
        :param: kernel: Instance of a KernelFunction class
        """
        if isinstance(x1, cs.MX) or isinstance(x2, cs.MX):
            cov_mat = cs.MX.zeros((x1.shape[0], x2.shape[0]))
            for i in range(x1.shape[0]):
                #a = .reshape(1,1)
                a = cs.reshape(x1[i,:], 1, x1.shape[1])
                for j in range(x2.shape[0]):

                    #b = x2[j,:].reshape(-1,1)
                    b = cs.reshape(x2[j,:], 1, x2.shape[1])
                    cov_mat[i,j] = kernel(a,b)
            return cov_mat


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


