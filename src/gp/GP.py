 # 
 # This file is part of the mpc_quad_ros distribution (https://github.com/smidmatej/mpc_quad_ros).
 # Copyright (c) 2023 Smid Matej.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #
 
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
        
    def __call__(self, x1 : float, x2 : float) -> float:
        """
        Calculate the value of the kernel function given 2 input vectors
        :param: x1: float or cs.MX of dimension 1
        :param: x2: float or cs.MX of dimension 1
        :return: k(x1,x2) float or cs.MX of dimension 1
        """
        assert self.kernel_type == "SEK", f"Kernel type is not of type SEK, kernel_type={self.kernel_type}"


        if (isinstance(x1, float) and isinstance(x2, float)) or (isinstance(x1, int) and isinstance(x2, int)):
            # Scalar implementation
            return float(self.sigma_f**2 * np.exp(-1/2*(x1-x2) * np.linalg.inv(self.L*self.L) * (x1-x2)))


        elif isinstance(x1, cs.MX) or isinstance(x2, cs.MX):
            # Casadi implementation
            # Use casadi if either x1 or x2 is a casadi object
            dif = x1-x2
            return self.sigma_f**2 * np.exp(-1/2* cs.mtimes(cs.mtimes(dif, np.linalg.inv(self.L*self.L)), dif.T))
        else:
            raise NotImplementedError("Only numpy and casadi are supported. Is the input a numpy array or casadi MX?")



    def __str__(self):
        return f"L = {self.L}, \n\r Sigma_f = {self.sigma_f}"


    
    
class GP:
    def __init__(self, X : np.array, y : np.array, covariance_function=KernelFunction, theta=[1,1,1]) -> None:
        """
        The whole contructor is in this method. This method is called when contructing this class and after self.fit()
        
        :param: X: np.array of n samples with dimension d. Input samples for regression
        :param: y: np.array of n samples with dimension d. Output samples for regression
        :param: covariance_function: Reference to a KernelFunction
        :param: theta: np.array of hyperparameters
        """
        if X is not None and y is not None and covariance_function is not None and theta is not None:
            self.initialize(X, y, covariance_function, theta)

        
    def initialize(self, X : np.ndarray, y : np.ndarray, covariance_function, theta) -> None:
        """
        The whole contructor is in this method. This method is called when contructing this class and after self.fit()
        
        :param: X: (n,) np.ndarray of n samples. Input samples for regression
        :param: y: (n,) np.ndarray of n samplesd. Output samples for regression
        :param: covariance_function: Reference to a KernelFunction
        :param: theta: np.ndarray of hyperparameters
        """

        assert X.ndim == 1, f"X.ndim = {X.ndim}, X.ndim must be 1"
        assert y.ndim == 1, f"y.ndim = {y.ndim}, y.ndim must be 1"

        if X is None or y is None:
            # No training data given, gp will only provide prior predictions
            self.n_train = 0
            self.X_dim = 1 # this needs to be set in a general way for prediction from prior
        else:
            self.n_train = X.shape[0]
            self.X_dim = 1


        self.X = X
        self.y = y
        self.covariance_function = covariance_function
        
        self.theta=theta
        self.kernel = covariance_function(L=np.eye(self.X_dim)*theta[0], sigma_f=theta[-2])
        
        self.noise = theta[-1]
        
        if isinstance(X, cs.MX):
            self.cov_matrix_of_input_data = self.calculate_covariance_matrix(X, X, self.kernel) \
                                            + (self.noise+1e-7)*np.identity(self.n_train)

            # Symbolic matrix inverse using linear system solve, since cs does not have native matrix inverse method
            self.inv_cov_matrix_of_input_data = cs.solve(self.cov_matrix_of_input_data, cs.MX.eye(self.cov_matrix_of_input_data.size1()))

        else:
            self.inv_cov_matrix_of_input_data = np.linalg.inv(
                self.calculate_covariance_matrix(X, X, self.kernel) \
                + (self.noise+1e-7)*np.identity(self.n_train))
        



    def predict(self, X_star, cov=False, var=False, std=False):
        """
        Evaluate the posterior mean m(x) and covariance sigma for supplied values of X_star
        :param X_star: (m,) np.ndarray of values to evaluate the posterior mean and covariance for, where m is the number of evaluation points
        """


        ### TODO: Add std and variance to casadi prediction ###
        sigma_k = self.calculate_covariance_matrix(self.X, X_star, self.kernel)
        sigma_kk = self.calculate_covariance_matrix(X_star, X_star, self.kernel)


        if self.n_train == 0:
            mean_at_values = np.zeros((X_star.shape[0],1))
        
            cov_matrix = sigma_kk
        
        else:
            if isinstance(X_star, cs.MX):
                mean_at_values = cs.mtimes(sigma_k.T, cs.mtimes(self.inv_cov_matrix_of_input_data, self.y))
                
                cov_matrix = np.eye(1) # TODO: Calculate covariance matrix for prediction if needed

               
            elif isinstance(X_star, np.ndarray):
                assert X_star.ndim == 1, "X_star must be a 1D np.ndarray"
                mean_at_values = sigma_k.T.dot(
                                        self.inv_cov_matrix_of_input_data.dot(
                                            self.y)) 

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
        

        
    def draw_function_sample(self, X_star, n_sample_functions=1):
        """ 
        Draw a function from the current distribution evaluated at X_star.
        
        :param X_star: np vector to evaluate function at
        :param n_sample_functions: allows for multiple sample draws from the same distribution
        """

        mean_at_values, cov_matrix = self.predict(X_star, cov=True)
        y_sample = np.random.multivariate_normal(mean=mean_at_values.ravel(), cov=cov_matrix, size=n_sample_functions)
        return y_sample
        
    def fit(self) -> None:
        """ 
        Uses the negative log likelyhood function to maximize likelyhood by varying the hyperparameters theta
        """
        
        #print(self.y.shape)
        low_bnd = 0.01
        #bnds = tuple([(low_bnd, None) for i in range(self.X.shape[1])]) + ((low_bnd, None), (low_bnd, None))
        bnds = ((low_bnd, None), (low_bnd, None), (low_bnd, None))
        #print('Maximizing the likelyhood function for GP')
        #print(f'Hyperparameters before optimization = {self.theta}')
        
        sol_min = minimize(self.nll, x0=self.theta, method='L-BFGS-B', bounds=bnds)
        theta_star = sol_min.x
        
        # Ammounts to recreating all relevant contents of this class
        self.initialize(self.X, self.y, self.covariance_function, theta_star)

        #print('Optimization done')
        #print(f'Hyperparameters after optimization = {self.theta}')

    def jacobian(self, z : cs.MX) -> cs.Function:
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
        

    def nll(self, theta : np.array) -> float:
        """
        Negative log likelyhood of the prediction using the theta hyperparameters.
        Numerically more stable implementation of Eq. (11)
        as described in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section 2.2, Algorithm 2.1.
        
        :param: theta: Hyperparameter np.array
        """

        # Kernel function k(x1,x2)
        k = self.covariance_function(L=np.eye(self.X_dim)*theta[0], sigma_f=theta[-2])
        
        # Evaluate k(x,x) over all combinations of x1 and x2
        K = self.calculate_covariance_matrix(self.X, self.X, k) + \
                (theta[-1]+1e-7)*np.identity(self.X.shape[0])
        
        L = cholesky(K)

        S1 = solve_triangular(L, self.y, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)


        neg_log_lklhd = (np.sum(np.log(np.diagonal(L))) + \
                       0.5 * self.y.T.dot(S2) + \
                       0.5 * self.X.shape[0] * np.log(2*np.pi)).flatten()
        return neg_log_lklhd



    def __str__(self):
        theta_string = '[' + ', '.join([f'{theta:.2f}' for theta in self.theta]) +']'
        return f"Theta: {theta_string}"




    @staticmethod
    def calculate_covariance_matrix(x1 : np.array, x2 : np.array, kernel) -> np.array:
        """
        Fills in a matrix with k(x1[i,:], x2[j,:])
        
        :param: x1: (n,) np.ndarray, where n is the number of samples to evaluate for
        :param: x2: (m,) np.ndarray, where m is the number of samples samples to evaluate for
        :param: kernel Instance of a KernelFunction class
        :return: (n,m) np.ndarray with k(x1[i], x2[j]) for all i and j
        """

        
        if isinstance(x1, cs.MX) or isinstance(x2, cs.MX):
            # Casadi implementation
            if isinstance(x1, np.ndarray):
                x1 = x1.reshape((x1.shape[0],1))
            if isinstance(x2, np.ndarray):
                x2 = x2.reshape((x2.shape[0],1))

            cov_mat = cs.MX.zeros((x1.shape[0], x2.shape[0]))
            for i in range(x1.shape[0]):

                a = cs.reshape(x1[i,:], 1, x1.shape[1])

                for j in range(x2.shape[0]):

                    b = cs.reshape(x2[j,:], 1, x2.shape[1])
                    cov_mat[i,j] = kernel(a,b)

            return cov_mat
        else:

            # Numpy implementation
            assert x1.ndim == 1, "x1 must be a 1D array"
            assert x2.ndim == 1, "x2 must be a 1D array"

            cov_mat = np.empty((x1.shape[0], x2.shape[0]))*np.NaN # Initialize the covariance matrix with NaNs
            
            # for all combinations calculate the kernel
            for i in range(x1.shape[0]):
                for j in range(x2.shape[0]):
                    cov_mat[i,j] = kernel(x1[i],x2[j])

            return cov_mat




    @staticmethod
    def save(gp:"GP", save_path:str) -> None:
        """
        Saves the gp to the specified save_path as a pickle file. Must be re-loaded with the load function
        :param gp: GP instance
        :param save_path: absolute save_path to save the gp to
        """

        saved_vars = {
            "kernel_params": gp.kernel.params,
            "kernel_type": gp.kernel.kernel_type,
            "X": gp.X,
            "y": gp.y,
            "theta": gp.theta,
            "X_dim": gp.X_dim,
        }

        save_path = save_path + '.gp'
        with open(save_path, 'wb') as f:
            joblib.dump(saved_vars, f)
        
    @staticmethod
    def load(load_path:str) -> "GP":
        """
        Load a pre-trained GP regressor
        :param load_path: path to pkl file with a dictionary with all the pre-trained matrices of the GP regressor
        """
        data_dict = joblib.load(load_path)

        gp = GP(data_dict['X'], data_dict['y'], KernelFunction, data_dict['theta'])
        #self.initialize(data_dict['X'], data_dict['y'], KernelFunction, data_dict['theta'])
        return gp
        


