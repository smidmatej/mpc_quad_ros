 # 
 # This file is part of the RGP distribution (https://github.com/smidmatej/RGP).
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
from scipy.linalg import sqrtm
import casadi as cs
import joblib


class RBF:
    """
    Radial Basis Function kernel function k(x1,x2) = sigma_f**2 * exp(-1/2*(x1-x2).T.dot(L.dot(L)).dot(x1-x2))
    """
    
    def __init__(self, L : np.array = np.eye(1), sigma_f : float = 1) -> None:
        """
        Contructor of the RBF function k(x1,x2).
        
        :param: L: Square np.array of dimension d x d. Defines the length scale of the kernel function
        :param: sigma_f: Scalar value used to linearly scale the amplidude of the k(x,x)
        """
        self.L = L
        self.sigma_f = sigma_f
        

    def __call__(self, x1 : float, x2 :  float) -> float:
        """
        Calculate the value of the kernel function given 2 input floats or cs.DM. 
        :param: x1: float or cs.MX
        :param: x2: float or cs.MX
        """

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


    def calculate_covariance_matrix(self, x1 : np.array, x2 : np.array) -> np.array:
        """
        Fills in a matrix with k(x1[i,:], x2[j,:])
        :param: x1: (n,) np.array or cs.MX, where n is the number of samples
        :param: x2: (n,) np.array or cs.MX, where n is the number of samples
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
                    cov_mat[i,j] = self.__call__(a,b)

            return cov_mat
        else:
            # Numpy implementation
            assert x1.ndim == 1, "x1 must be a 1D array"
            assert x2.ndim == 1, "x2 must be a 1D array"

            cov_mat = np.empty((x1.shape[0], x2.shape[0]))*np.NaN # Initialize the covariance matrix with NaNs
            
            # for all combinations calculate the kernel
            for i in range(x1.shape[0]):
                for j in range(x2.shape[0]):
                    cov_mat[i,j] = self.__call__(x1[i],x2[j])

            return cov_mat

    def __str__(self):
        return f"L = {self.L}, \n\r Sigma_f = {self.sigma_f}"
        

class RGP:
    def __init__(self, X : np.ndarray, y_ : np.ndarray, C : np.ndarray = None, theta : list = [1.0,0.1,0.1]) -> None:
        """
        Recursive Gaussian Process (RGP) class. Initialize with n basis vectors and n measurements and hyperparameters theta. 
        Recursively update the RGP with new measurements using regress(). Update and predict the hyperparameters using learn().
        :param: X: (n,) np.ndarray, where n is the number of basis vectors
        :param: y_: (n,) np.ndarray, where n is the number of basis vectors
        :param: C: (n,n) np.ndarray, where n is the number of basis vectors. If None, C is initialized as the identity matrix
        :theta: theta: list of hyperparameters [L, sigma_f, sigma_n]
        """

        # TODO: THIS CLASS CAN IMPLEMENT FIT THE SAME WAY AS GP, SINCE THE GP.FIT TAKES JUST THE BASIS VECTORS AND THEIR RESPONSE
        # but I dont know the response, the reason to use rgp is to not have to have a response before using gp
        assert X.ndim == 1, "X must be a 1D array"
        assert y_.ndim == 1, "y_ must be a 1D array"
        assert X.shape[0] == y_.shape[0], "X and y_ must have the same number of rows"
        assert len(theta) == 3, "theta must be a list of 3 hyperparameters [L, sigma_f, sigma_n]"
        if C is not None:
            assert C.shape[0] == C.shape[1], "C must be a square matrix"
            assert C.shape[0] == X.shape[0], "C must have the same number of rows as X and y_"

        self.X = X
        self.y_ = y_
        
        # L and sigma_f are the hyperparameters of the RBF kernel function, they are not properties of the RGP
        L = np.eye(1) * theta[0]
        #L = np.diag(theta[0])
        sigma_f = theta[1] # RBF
        self.sigma_n = theta[2] # Noise variance

        # Mean function m(x) = 0
        self.K = RBF(L=L, sigma_f=sigma_f) # Kernel function

        # WARNING: Dont confuse the estimate g at X with the estimate g_t at X_t 
        # p(g|y_t-1)
        self.mu_g_t = y_ # The a priori mean is the measurement with no y_t 
        if C is not None:
            self.C_g_t = C # Use the provided covariance matrix
        else:
            self.C_g_t = self.K.calculate_covariance_matrix(X, X) + self.sigma_n**2 * np.eye(self.X.shape[0]) # The a priori covariance is the covariance with no y_t
        
        # Hyperparameter estimates for RGP*
        # np.log to transform L into strictly positive values for training, inverse transformation is done at the end of learning
        #self.mu_eta_t = np.concatenate([np.log(np.diagonal(L)), [np.log(sigma_f)], [np.log(self.sigma_n)]])  # The a priori mean of the hyperparameters is the hyperparameters
        self.mu_eta_t = np.concatenate([np.diagonal(L), [sigma_f], [self.sigma_n]])  # The a priori mean of the hyperparameters is the hyperparameters
        self.C_eta_t = np.eye(self.mu_eta_t.shape[0]) # The a priori covariance of the hyperparameters is the identity matrix
        
        # Cross-covariance between the basis vectors and the hyperparameters
        self.C_g_eta_t = np.zeros((self.X.shape[0], self.mu_eta_t.shape[0])) # The a priori covariance is zero

        # Precompute these since they do not change with regression (They change during learning, since the hyperparameters change)
        self.K_x = self.K.calculate_covariance_matrix(self.X, self.X) + self.sigma_n**2 * np.eye(self.X.shape[0]) # Covariance matrix over X
        self.K_x_inv = np.linalg.inv(self.K_x) # Inverse of the covariance matrix over X


        
    def get_theta(self) -> list:
        """
        Get the hyperparameters of the RGP
        :return: list of hyperparameters [L, sigma_f, sigma_n]
        """
        return [self.K.L, self.K.sigma_f, self.sigma_n]

    def predict(self, X_t_star : np.ndarray, cov : bool = False, var : bool = False, std : bool = False, return_Jt : bool = False) -> np.ndarray:
        """
        Predict the value of the response at X_t_star given the data GP.
        :param: X_t_star: (m,) np.array or cs.MX, where m is the number of points to predict
        :param: cov: Boolean value. If true, the covariance matrix of the prediction is calculated and returned as well
        :param: var: Boolean value. If true, the variance of the prediction is calculated and returned as well
        :param: std: Boolean value. If true, the standard deviation of the prediction is calculated and returned as well
        :param: return_Jt: Boolean value. If true, the gain matrix Jt is returned as well INTERNAL USE ONLY
        """
        assert isinstance(X_t_star, np.ndarray) or isinstance(X_t_star, cs.MX), "X_t_star must be a np.array or cs.MX"



        if isinstance(X_t_star, cs.MX):
            # Casadi implementation
            K_x_star = self.K.calculate_covariance_matrix(X_t_star, self.X)
            Jt = cs.mtimes(K_x_star, self.K_x_inv) # Gain matrix
            mu_p_t = cs.mtimes(Jt, self.mu_g_t) # The a posteriori mean of p(g_t|y_t)

            if cov or var or std:
                K_x_star_star = self.K.calculate_covariance_matrix(X_t_star, X_t_star)
                # Is self.K.calculate_covariance_matrix(X_t_star, self.X).T = self.K.calculate_covariance_matrix(self.X, X_t_star) ?
                B =  K_x_star_star - cs.mtimes(Jt, self.K.calculate_covariance_matrix(self.X, X_t_star))
                C_p_t = B + cs.mtimes(Jt, cs.mtimes(self.C_g_t, Jt.T)) # The a posteriori covariance of p(g_t|y_t)
                var_p_t = cs.diag(C_p_t) # The variance of p(g_t|y_t)
                std_p_t = cs.sqrt(var_p_t) # The standard deviation of p(g_t|y_t)
                
        else:
            # Numpy implementation
            assert X_t_star.ndim == 1, "X_t_star must be a 1D array"

            Jt = self.K.calculate_covariance_matrix(X_t_star, self.X).dot(self.K_x_inv) # Gain matrix
            mu_p_t = Jt.dot(self.mu_g_t) # The a posteriori mean of p(g_t|y_t)

            mu_p_t = mu_p_t.ravel() # return as (m,)
            
            if cov or var or std:
                # Calculate and return the covariance matrix too
                K_x_star_star = self.K.calculate_covariance_matrix(X_t_star, X_t_star)
                B = K_x_star_star - Jt.dot(self.K.calculate_covariance_matrix(self.X, X_t_star)) # Covariance of p(g_t|g_)
                C_p_t = B + Jt.dot(self.C_g_t).dot(Jt.T) # The a posteriori covariance of p(g_t|y_t)
                var_p_t = np.diag(C_p_t) # The variance of p(g_t|y_t)
                std_p_t = np.sqrt(var_p_t) # The standard deviation of p(g_t|y_t)
            
        if return_Jt:
            if cov:
                return mu_p_t, C_p_t, Jt
            elif var:
                return mu_p_t, var_p_t, Jt
            elif std:
                return mu_p_t, std_p_t, Jt
            else:
                return mu_p_t, Jt
        else:
            if cov:
                return mu_p_t, C_p_t
            elif var:
                return mu_p_t, var_p_t
            elif std:
                return mu_p_t, std_p_t
            else:
                return mu_p_t





    def predict_using_y(self, X_t_star : np.ndarray, y : np.ndarray, cov : bool = False, var : bool = False, std : bool = False, return_Jt : bool = False) -> np.ndarray:
        """
        Predict the value of the response at X_t_star given the vector y corresponding to the response of the regressed function at X.
        :param: X_t_star: (m,) np.array or cs.MX, where m is the number of points to predict
        :param: y: (n,) np.array or cs.MX, where n is the number of basis vectors. The response at the basis points
        :param: cov: Boolean value. If true, the covariance matrix of the prediction is calculated and returned as well
        :param: var: Boolean value. If true, the variance of the prediction is calculated and returned as well
        :param: std: Boolean value. If true, the standard deviation of the prediction is calculated and returned as well
        :param: return_Jt: Boolean value. If true, the gain matrix Jt is returned as well INTERNAL USE ONLY
        """

        assert isinstance(X_t_star, np.ndarray) or isinstance(X_t_star, cs.MX), "X_t_star must be a np.array or cs.MX"
        assert isinstance(y, np.ndarray) or isinstance(y, cs.MX), "y must be a np.array or cs.MX"


        if isinstance(X_t_star, cs.MX) or isinstance(y, cs.MX):
            # Casadi implementation
            K_x_star = self.K.calculate_covariance_matrix(X_t_star, self.X)
            Jt = cs.mtimes(K_x_star, self.K_x_inv) # Gain matrix
            mu_p_t = cs.mtimes(Jt, y) # The a posteriori mean of p(g_t|y_t)

            if cov or var or std:
                K_x_star_star = self.K.calculate_covariance_matrix(X_t_star, X_t_star)
                # Is self.K.calculate_covariance_matrix(X_t_star, self.X).T = self.K.calculate_covariance_matrix(self.X, X_t_star) ?
                B =  K_x_star_star - cs.mtimes(Jt, self.K.calculate_covariance_matrix(self.X, X_t_star))
                C_p_t = B + cs.mtimes(Jt, cs.mtimes(self.C_g_t, Jt.T)) # The a posteriori covariance of p(g_t|y_t)
                var_p_t = cs.diag(C_p_t) # The variance of p(g_t|y_t)
                std_p_t = cs.sqrt(var_p_t) # The standard deviation of p(g_t|y_t)

        elif isinstance(X_t_star, np.ndarray) and isinstance(y, np.ndarray):
            # Numpy implementation
            assert X_t_star.ndim == 1, "X_t_star must be a 1D array"

            Jt = self.K.calculate_covariance_matrix(X_t_star, self.X).dot(self.K_x_inv) # Gain matrix
            mu_p_t = Jt.dot(y) # The a posteriori mean of p(g_t|y_t)

            mu_p_t = mu_p_t.ravel() # return as (m,)
            
            if cov or var or std:
                # Calculate and return the covariance matrix too
                K_x_star_star = self.K.calculate_covariance_matrix(X_t_star, X_t_star)
                B = K_x_star_star - Jt.dot(self.K.calculate_covariance_matrix(self.X, X_t_star)) # Covariance of p(g_t|g_)
                C_p_t = B + Jt.dot(self.C_g_t).dot(Jt.T) # The a posteriori covariance of p(g_t|y_t)
                var_p_t = np.diag(C_p_t) # The variance of p(g_t|y_t)
                std_p_t = np.sqrt(var_p_t) # The standard deviation of p(g_t|y_t)
        else:
            raise ValueError("Either X_t_star or y must be a cs.MX object, or both must be np.array objects")
            
        if return_Jt:
            if cov:
                return mu_p_t, C_p_t, Jt
            elif var:
                return mu_p_t, var_p_t, Jt
            elif std:
                return mu_p_t, std_p_t, Jt
            else:
                return mu_p_t, Jt
        else:
            if cov:
                return mu_p_t, C_p_t
            elif var:
                return mu_p_t, var_p_t
            elif std:
                return mu_p_t, std_p_t
            else:
                return mu_p_t


    def regress(self, Xt : np.ndarray, yt : np.ndarray) -> np.ndarray:
        """
        Modify the RGP mean and covariance to account for new data Xt and yt.
        :param: Xt: (k,) np.array, where k is the number of new data points
        :param: yt: (k,) np.array, where k is the number of new data points
        """
        assert Xt.ndim == 1, "Xt must be a 1D array"
        assert yt.ndim == 1, "yt must be a 1D array"
        assert Xt.shape == yt.shape, "Xt and yt must have the same shape"

        # ------ New data received -> step the memory forward ------
        self.mu_g_t_minus_1 = self.mu_g_t # The a priori mean is the estimate of g at X_
        self.C_g_t_minus_1 = self.C_g_t

        # ------ Inference step ------
        # Infer the a posteriori distribution of p(g_t|y_t) (the estimate of g_t at X_t)
        mu_p_t, C_p_t, Jt = self.predict(Xt, cov = True, return_Jt = True)


        # ------ Update step ------
        # Update the a posteriori distribution of p(g_|y_t) (the estimate of g at X)
        G_tilde_t = self.C_g_t_minus_1.dot(Jt.T).dot(
                np.linalg.inv(
                    C_p_t + self.sigma_n**2 * np.eye(Xt.shape[0]))) # Kalman gain
        self.mu_g_t = self.mu_g_t_minus_1 + G_tilde_t.dot(yt - mu_p_t) # The a posteriori mean of p(g_|y_t)
        self.C_g_t = self.C_g_t_minus_1 - G_tilde_t.dot(Jt).dot(self.C_g_t_minus_1) # The a posteriori covariance of p(g_|y_t)

        return self.mu_g_t, self.C_g_t

    def learn(self, Xt : np.ndarray, yt : np.ndarray) -> np.ndarray:
        """
        Performs both the updating of the basis vectors, but also the hyperparameter optimization
        """
        
        n_eta = self.mu_eta_t.shape[0] # State dimension of eta
        n_g = self.mu_g_t.shape[0] # State dimension of g
        n_g_t = yt.shape[0] # State dimension of g_t
        n_p = n_g + n_eta + n_g_t # State dimension of p

        assert n_g_t == 1, "Only one-dimensional regression is supported"
        assert Xt.shape[0] == 1, "Only one-dimensional regression is supported"

        # ------ New data received -> step the memory forward ------
        self.mu_g_t_minus_1 = self.mu_g_t # The a priori mean is the estimate of g at X_
        self.C_g_t_minus_1 = self.C_g_t

        self.mu_eta_t_minus_1 = self.mu_eta_t
        self.C_eta_t_minus_1 = self.C_eta_t

        self.C_g_eta_t_minus_1 = self.C_g_eta_t


        # ------! Inference step !------

        Jt = self.K.calculate_covariance_matrix(Xt, self.X).dot(self.K_x_inv) # Gain matrix (same as in regression)
        assert Jt.shape[1] == n_g, "Jt.shape[1] != n_g"
        B = self.K.calculate_covariance_matrix(Xt, Xt) - Jt.dot(self.K.calculate_covariance_matrix(self.X, Xt)) # Covariance of p(g_t|g_)
        St = self.C_g_eta_t_minus_1.dot(np.linalg.inv(self.C_eta_t_minus_1))

        # At is a function of Jt which is a function of eta (nonlinear function)
        At = np.asarray(np.bmat([
                        [np.eye(n_g), np.zeros((n_g, n_eta))],
                        [np.zeros((n_eta, n_g)), np.eye(n_eta)],
                        [Jt, np.zeros((1, n_eta))]])) # I prefer using np arrays instead of np matrices

        mu_w_t = np.zeros((n_p, )) # This is zero because of the zero mean function of GP. Should be nonzero in general
        C_w_t = np.asarray(np.bmat([
            [np.zeros((n_g, n_g)), np.zeros((n_g, n_eta)), np.zeros((n_g, n_g_t))],
            [np.zeros((n_eta, n_g)), np.zeros((n_eta, n_eta)), np.zeros((n_eta, n_g_t))],
            [np.zeros((n_g_t, n_g)), np.zeros((n_g_t, n_eta)), B]]))
        
        assert mu_w_t.shape[0] == At.shape[0], "mu_w_t.shape[0] != At.shape[0]"
        assert mu_w_t.shape[0] == C_w_t.shape[0], "mu_w_t.shape[0] != C_w_t.shape[0]"

        
        # ------ Unscented transform ------
        w, eta_hat = self.__draw_sigma_points(self.mu_eta_t_minus_1, self.C_eta_t_minus_1)
        s = w.shape[0] # Number of sigma points
        
        # p = [g, eta, g_t]
        mu_p_i = np.empty((s, n_p)) # Allocate memory
        C_p_i = np.empty((s, n_p, n_p)) # Allocate memory
        
        mu_p_t = np.zeros((n_p, )) # Allocate memory
        C_p_t = np.zeros((n_p, n_p)) # Allocate memory
        for i in range(s):
            
            # --------- Individual predictions from sigma points ---------
            # Transform the sigma points
            mu_p_i[i,:] = At.dot(np.concatenate([
                    self.mu_g_t_minus_1.ravel() + St.dot(eta_hat[i,:] - self.mu_eta_t_minus_1),
                    eta_hat[i,:]]
                    , axis=0)).ravel() + mu_w_t

            
            tmp_matrix = np.bmat([[self.C_g_t_minus_1 - St.dot(self.C_g_eta_t_minus_1.T), np.zeros((n_g, n_eta))],[np.zeros((n_eta, n_g)), np.zeros((n_eta, n_eta))]])
            C_p_i[i,:,:] = At.dot(np.asarray(tmp_matrix)).dot(At.T) + C_w_t
        
            # --------- Combine individual predictions ---------
            # Cummulative sum
            mu_p_t += w[i] * mu_p_i[i,:]
            C_p_t += w[i] * (np.outer(mu_p_i[i,:] - mu_p_t, mu_p_i[i,:] - mu_p_t) + C_p_i[i,:,:])

        
        # ------! Update step !------

        # Decomposition of mu_p_t into observable and unobservable parts
        

        # Observable part
        # o = [sigma_n, g_t]
        mu_o_t = mu_p_t[n_g + n_eta - 1:] # sigma_n is on index n_g+n_eta-1 and is last of eta, everything after is is g_t
        C_o_t = C_p_t[n_g + n_eta - 1:, n_g + n_eta - 1:]

        # Unobservable part
        # u = [g, eta-]  (eta- is eta without the last element, sigma_n)
        mu_u_t_minus_1 = mu_p_t[:n_g + n_eta - 1]
        C_u_t_minus_1 = C_p_t[:n_g + n_eta - 1, :n_g + n_eta - 1]
        # Covariance between observable and unobservable parts
        C_ou_t = C_p_t[n_g + n_eta - 1:, :n_g + n_eta - 1]

        # ------ Update observable state ------
        mu_y_t = mu_o_t[1:] # g_t (without sigma_n)
        C_y_t = C_o_t[1:, 1:] + C_o_t[0, 0] + mu_o_t[0]**2 

        C_o_y_t = C_o_t[:, 1:] # Covariance between observable part and y_t
        
        Gt = C_o_y_t.dot(np.linalg.inv(C_y_t)) # Kalman gain

        # Updated observable part
        # e = [sigma_n, g_t]
        mu_e_t = mu_o_t + Gt.dot(yt - mu_y_t)
        C_e_t = C_o_t - Gt.dot(C_y_t).dot(Gt.T)

        # ------ Update joint state ------
        # This update has the same structure as the Rauch-Tung-Striebel smoother according to the article
        Lt = C_ou_t.T.dot(np.linalg.inv(C_o_t)) # Kalman gain

        mu_u_t = mu_u_t_minus_1 + Lt.dot(mu_e_t - mu_o_t)
        C_u_t = C_u_t_minus_1 + Lt.dot(C_e_t - C_o_t).dot(Lt.T)


        # u = [g, eta-]
        # e = [sigma_n, g_t]
        # z = [g, eta]
        

        h = np.zeros((mu_e_t.shape[0],)) # Select first element of mu_e_t
        h[0] = 1
        # sigma_n = h.dot(mu_e_t)
        
        mu_z_t = np.concatenate([mu_u_t, [h.dot(mu_e_t)]], axis=0)
        C_z_t = np.asarray(np.bmat([
            [C_u_t, (Lt.dot(C_e_t).dot(h.T)).reshape((-1,1))],
            [(h.dot(C_e_t).dot(Lt.T)).reshape((1,-1)), np.array([h.dot(C_e_t).dot(h.T)]).reshape(1,1)]]))


        self.mu_g_t = mu_z_t[:n_g]
        self.C_g_t = C_z_t[:n_g, :n_g]

        self.mu_eta_t = mu_z_t[n_g:]
        self.C_eta_t = C_z_t[n_g:, n_g:]

        #breakpoint()
        # Use the updated hyperparameters
        self.K.L = np.diag([np.exp(self.mu_eta_t[0])])
        self.K.sigma_f = np.exp(self.mu_eta_t[1])
        self.sigma_n = np.exp(self.mu_eta_t[2])

        self.K.L = np.diag([self.mu_eta_t[0]])
        self.K.sigma_f = self.mu_eta_t[1]
        self.sigma_n = self.mu_eta_t[2]

        # These "precomputed" matrices need to be updated with the new hyperparameters as well
        self.K_x = self.K.calculate_covariance_matrix(self.X, self.X) + self.sigma_n**2 * np.eye(self.X.shape[0]) # Covariance matrix over X
        self.K_x_inv = np.linalg.inv(self.K_x) # Inverse of the covariance matrix over X

        return mu_z_t, C_z_t


    def __draw_sigma_points(self, mu : np.ndarray, C : np.ndarray) -> np.ndarray:
        """
        Draws sigma points from a Gaussian distribution using the unscented transform
        """
        # --------- Unscented transform ---------

        n = mu.shape[0] # State dimension of mu

        w = np.empty((2*n+1,))
        x = np.empty((2*n+1, n)) # 2n+1 sigma points in R^n
        w[0] = 0.5
        x[0,:] = mu
        

        for i in range(n):
            # index 1 to n
            x[i+1,:] = mu + sqrtm(n/(1-w[0]) * C)[:,i] # ith collumn of the matrix sqrt
            x[i+1+n,:] = mu - sqrtm(n/(1-w[0]) * C)[:,i] # ith collumn of the matrix sqrt
            
            w[i+1] = (1-w[0])/(2*n)
            w[i+1+n] = (1-w[0])/(2*n)
        
        return w, x

    @staticmethod
    def save(rgp:"RGP", save_path:str) -> None:
        """
        Saves the gp to the specified save_path as a pickle file. Must be re-loaded with the load function
        :param gp: GP instance
        :param save_path: absolute save_path to save the gp to
        """

        saved_vars = {
            "X": rgp.X,
            "y": rgp.y_,
            "theta": rgp.mu_eta_t,
        }
        save_path = save_path + '.rgp'
        with open(save_path, 'wb') as f:
            joblib.dump(saved_vars, f)
        
    @staticmethod
    def load(load_path:str) -> "RGP":
        """
        Load a pre-trained GP regressor
        :param load_path: path to pkl file with a dictionary with all the pre-trained matrices of the GP regressor
        """
        data_dict = joblib.load(load_path)

        rgp = RGP(data_dict['X'], data_dict['y'], theta=data_dict['theta'])
        #self.initialize(data_dict['X'], data_dict['y'], KernelFunction, data_dict['theta'])
        return rgp



