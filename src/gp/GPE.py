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
import matplotlib.pyplot as plt
import seaborn as sns
import rospy
try:
    from GP import GP
    from RGP import RGP
except ImportError:
    from gp.GP import GP
    from gp.RGP import RGP
import numpy as np
import casadi as cs
import os
import time
import matplotlib

class GPEnsemble:

    def __init__(self, gp_list : list, type : str) -> None:
        """
        Ensamble of (Recursive) Gaussian Process regressors. Holds a list of GP or RGP objects, one for each dimension of the output.
        :param number_of_dimensions: Number of dimensions of the output
        :param type: Type of GP to be used. 'GP' or 'RGP'
        """
        self.gp = gp_list
        self.type = type


    @classmethod
    def fromdims(cls, number_of_dimensions : int, type : str):
        gp_list = [None]*number_of_dimensions
        return cls(gp_list=gp_list, type=type)

    @classmethod
    def fromlist(cls, gp_list : list = []):
        """
        Creates a GPEnsemble from a list of GP objects
        :param gp_list: List of GP objects
        """
        if all([isinstance(g, GP) for g in gp_list]):
            type = 'GP'
        elif all([isinstance(g, RGP) for g in gp_list]):
            type = 'RGP'
        else:
            raise ValueError("All GP objects in the list must be of the same type")

        return cls(gp_list=gp_list, type=type)

    @classmethod
    def frombasisvectors(cls, X : list, y : list, C : list, theta : list):
        """
        Creates a GPEnsemble from a list of basis vectors and their corresponding y values
        :param X: List of np.ndarray with the basis vectors
        :param y: List of np.ndarray with the corresponding y values
        """
        assert len(X) == 3, "X must have length 3"
        assert len(y) == 3, "y must have length 3"
        assert len(C) == 3, "C must have length 3"
        assert len(theta) == 3, "theta must have length 3"
        
        type = 'RGP'
        gp_list = [None]*3
        for i in range(3):
            gp_list[i] = RGP(X[i], y[i], C[i], theta[i])


        return cls(gp_list=gp_list, type=type)

    @classmethod
    def fromdir(cls, filepath : str, type : str):
        """
        Loads GPs from path and adds them to the GPEnsemble
        :param: path: folder path to load the models. Models will be loaded from inside the folder with the name model_x, model_y, model_z
        """
        print("Loading GPEnsemble from path: ", filepath)

        files_in_directory = os.listdir(filepath) # list all the files in the directory
        number_of_files = len(files_in_directory)
        print(f"Found {number_of_files} files in directory: {filepath}")
        print("Files: ", files_in_directory)

        gp_list = list()
        for file in files_in_directory:
            if file.endswith(".gp") and type == 'GP':
                path_to_file = os.path.join(filepath, file)
                gp_list.append(GP.load(path_to_file))
            elif file.endswith(".rgp") and type == 'RGP':
                path_to_file = os.path.join(filepath, file)
                gp_list.append(RGP.load(path_to_file))
                
        return cls(gp_list, type=type)

    @classmethod
    def fromemptybasisvectors(cls, X : 'list[np.ndarray]'):
        """
        Creates a GPEnsemble from a list of basis vectors and sets their corresponding y values to zero
        :param X: List of np.ndarray with the basis vectors
        """
        assert len(X) == 3, "X must have length 3"
        
        type = 'RGP'
        gp_list = [None]*3
        for i in range(3):
            y = np.zeros((X[i].shape[0], ))
            gp_list[i] = RGP(X[i], y)


        return cls(gp_list=gp_list, type=type)
    
    @classmethod
    def fromrange(cls, x_min_max : 'list[tuple[float, float]]', n_basis : 'list[int]'):
        """
        Creates a GPEnsemble from a list of basis vectors and sets their corresponding y values to zero
        :param x_min_max: List of tuples with the minimum and maximum values of the basis vectors
        :param n_basis: List of integers with the number of basis vectors
        """
        assert len(x_min_max) == 3, "X must have length 3"
        assert len(n_basis) == 3, "n_basis must have length 3"
        
        type = 'RGP'
        gp_list = [None]*3
        for i in range(3):

            X_basis = np.linspace(x_min_max[i][0], x_min_max[i][1], n_basis[i])
            #breakpoint()
            y = np.zeros((X_basis.shape[0], ))
            gp_list[i] = RGP(X_basis, y)


        return cls(gp_list=gp_list, type=type)



    def get_theta(self) -> list:
        """
        Returns the hyperparameters of the GPEnsemble
        :return: List of hyperparameters
        """
        theta = [None]*len(self.gp)
        for n in range(len(self.gp)):
            theta[n] = self.gp[n].get_theta()

        return theta

    def predict(self, X_t : list, std : bool = False) -> list:
        """
        Predicts the output of the GPEnsemble at X_t
        :param X_t: list of np.ndarray to the GPEnsemble
        :param std: If True, returns the standard deviation of the prediction
        :return: Prediction of the GPEnsemble at X_t
        """
        ### TODO: Add std and variance to casadi prediction ###
        assert len(X_t) == len(self.gp), "X_t must be a list of np.ndarray with the same length as the number of GPs in the GPEnsemble"
        assert all([isinstance(X_t[n], np.ndarray) or isinstance(X_t[n], cs.MX) for n in range(len(X_t))]), "X_t must be a list of np.ndarray or casadi.MX"

        mu_dim = [None]*len(self.gp)
        std_dim = [None]*len(self.gp)
        rospy.logwarn(f"X_t[0]: {type(X_t[0])}")
        # Do the predictions
        for n in range(len(self.gp)):
            if std:
                mu_dim[n], std_dim[n]  = self.gp[n].predict(X_t[n], std=std)
            else:
                mu_dim[n] = self.gp[n].predict(X_t[n], std=std)


        # Output formatting
        if all([isinstance(mu_dim[n], np.ndarray) for n in range(len(mu_dim))]):
            mu = mu_dim
            if std:
                std = std_dim
                return mu, std
        elif all([isinstance(mu_dim[n], cs.MX) for n in range(len(mu_dim))]):
            mu = cs.horzcat(*mu_dim)
            if std:
                std = cs.horzcat(*std_dim)
                return mu, std
        else:
            raise ValueError("Output mu must be a list of np.ndarray or casadi.MX")
        
        return mu

    def predict_using_y(self, X_t : list, y : list, std : bool = False) -> np.ndarray:
        """
        Predicts the output of the GPEnsemble at X_t given the output y of the GPEnsemble
        :param X_t: list of np.ndarray or cs.MX to the GPEnsemble
        :param y: list of np.ndarray or cs.MX of the output of the GPEnsemble
        :param std: If True, returns the standard deviation of the prediction
        :return: Prediction of the GPEnsemble at X_t
        """
        ### TODO: Add std and variance to casadi prediction ###
        assert len(X_t) == len(self.gp), "X_t must be a list of np.ndarray with the same length as the number of GPs in the GPEnsemble"
        assert len(y) == len(self.gp), "y must be a list of np.ndarray with the same length as the number of GPs in the GPEnsemble"
        assert all([isinstance(X_t[n], np.ndarray) or isinstance(X_t[n], cs.MX) for n in range(len(X_t))]), "X_t must be a list of np.ndarray or casadi.MX"
        assert all([isinstance(y[n], np.ndarray) or isinstance(y[n], cs.MX) for n in range(len(y))]), "y must be a list of np.ndarray or casadi.MX"

        mu_dim = [None]*len(self.gp)
        std_dim = [None]*len(self.gp)

        # Do the predictions
        for n in range(len(self.gp)):
            if std:
                mu_dim[n], std_dim[n]  = self.gp[n].predict_using_y(X_t[n], y[n], std=std)
            else:
                mu_dim[n] = self.gp[n].predict_using_y(X_t[n], y[n], std=std)
        
        # Output formatting
        if all([isinstance(mu_dim[n], np.ndarray) for n in range(len(mu_dim))]):
            mu = np.concatenate(mu_dim, axis=1)
            if std:
                std = np.concatenate(std_dim, axis=1)
                return mu, std
        elif all([isinstance(mu_dim[n], cs.MX) for n in range(len(mu_dim))]):
            mu = cs.horzcat(*mu_dim)
            if std:
                std = cs.horzcat(*std_dim)
                return mu, std
        else:
            raise ValueError("Output mu must be a list of np.ndarray or casadi.MX")
        
        return mu


    def regress(self, X_t : list, y_t : list) -> list:
        """
        Regress the RGP to the data X_t, y_t
        :param X_t: list of np.ndarray to the GPEnsemble
        :param y_t: list of np.ndarray to the GPEnsemble
        :return: Prediction of the GPEnsemble at X_t
        """
        ### TODO: Add std and variance to casadi prediction ###
        assert len(X_t) == len(self.gp), "X_t must be a list of np.ndarray with the same length as the number of GPs in the GPEnsemble"
        assert all([isinstance(X_t[n], np.ndarray)for n in range(len(X_t))]), "X_t must be a list of np.ndarray"
        assert len(y_t) == len(self.gp), "y_t must be a list of np.ndarray with the same length as the number of GPs in the GPEnsemble"
        assert all([isinstance(y_t[n], np.ndarray)for n in range(len(y_t))]), "y_t must be a list of np.ndarray"

        mu_dim = [None]*len(self.gp)
        C_dim = [None]*len(self.gp)

        # Do the regressions
        for n in range(len(self.gp)):
            mu_dim[n], C_dim[n]  = self.gp[n].regress(X_t[n], y_t[n])

        #breakpoint()
        # Output formatting
        #mu = np.concatenate(mu_dim, axis=1)
        #C = np.concatenate(C_dim, axis=1)
        return mu_dim, C_dim


        
    def fit(self) -> None:
        """
        Fits all GPs in the GPEnsemble
        """
        if self.type == 'RGP':
            raise NotImplementedError("RGP is not fitted with fit() method, use regress() instead")
        print("Fitting GPEnsemble")

        start_time = time.time()
        for gpr in self.gp:
            gpr.fit()

        print(f"Fitted GPEnsemble in {(time.time() - start_time):.2f} seconds")

    def jacobian(self, z):
        """
        Casadi symbolic jacobian of expression self.prediction with respect to z

        :param: z: Casadi symbolic vector expression n x d
        :return: Casadi function jacobian
        """

        if self.type == 'RGP':
            raise NotImplementedError("RGP does not have jacobian() method")

        assert z.shape[1] == len(self.gp), f"z needs to be n x d,  z.shape={z.shape}, GPE.number_of_dimensions={len(self.gp)}"

        f_jacobs = list()
        for col in range(len(self.gp)):
            f_jacobs.append(self.gp[col].jacobian(z[:,col]))
        return f_jacobs

    def save(self, path : str) -> None:
        """
        Runs GP.save() (or RGP.save()) for each GP (or RGP) in the GPEnsemble
        :param: path: folder path to save the models. Models will be saved inside the folder with the name gp_x, gp_y, gp_z (or rgp_x, rgp_y, rgp_z)
        """

        if os.path.exists(path):
            print(f"Saving models inside inside {path}")
        else:
            os.makedirs(path)
            print(f"Created folder {path}, saving models inside")

        xyz_name = ['mdl_x','mdl_y','mdl_z']
        assert len(self.gp) == len(xyz_name), f"GPEnsemble has {len(self.gp)} GPs, but xyz_name has {len(xyz_name)} names"

        for i_gp in range(len(self.gp)):
            path_with_name = os.path.join(path, xyz_name[i_gp])
            
            if self.type == 'GP':
                GP.save(self.gp[i_gp], path_with_name)
            elif self.type == 'RGP':
                RGP.save(self.gp[i_gp], path_with_name)
            else:
                raise ValueError(f"Unknown type {self.type}")
        




    # USE THE CLASSMETHOD TO LOAD THE GPEnsemble!
    '''

        def load(self, path : str) -> None:
        """
        Loads GPs from path and adds them to the GPEnsemble
        :param: path: folder path to load the models. Models will be loaded from inside the folder with the name model_x, model_y, model_z
        """
        print("Loading GPEnsemble from path: ", path)
        breakpoint()
        if self.type == 'RGP':
            xyz_name = ['rgp_x','rgp_y','rgp_z']
            for i_gp in range(len(xyz_name)):
                path_with_name = os.path.join(path, xyz_name[i_gp])
                # Create a new empty RGP and add it to GPE
                self.gp[i_gp] = RGP.load(path_with_name)

        elif self.type == 'GP':

            xyz_name = ['model_x','model_y','model_z']
            # GPE contains 3 GPs, one for each dimension
            for i_gp in range(len(xyz_name)):
                path_with_name = os.path.join(path, xyz_name[i_gp])
                # Create a new empty GP and add it to GPE
                self.gp[i_gp] = GP.load(path_with_name)


        #len(self.gp) = len(self.gp)
    '''



    def plot(self, z_train : np.array = None , y_train : np.array = None, filepath=None, show=True):
        

        assert z_train.ndim == 2, f"z_train needs to be n x d,  z_train.shape={z_train.shape}"
        assert y_train.ndim == 2, f"y_train needs to be n x d,  y_train.shape={y_train.shape}"
        assert z_train.shape[1] == len(self.gp), f"z_train needs to be n x d,  z_train.shape={z_train.shape}, GPE.number_of_dimensions={len(self.gp)}"
        assert y_train.shape[1] == len(self.gp), f"y_train needs to be n x d,  y_train.shape={y_train.shape}, GPE.number_of_dimensions={len(self.gp)}"
        
        # TODO: Why am I passing the z_train and y_train here? I can just use the data from the GPE

        z_query = [None]*len(self.gp)
        y_query = [None]*len(self.gp)
        std_query = [None]*len(self.gp)
        for dim in range(len(self.gp)):
            z_query[dim] = np.arange(-20,20,0.5) # TODO: This is a (n,) array, but the GP.predict() method expects a (n,1) array. FIX THE GP.predict() method
            y_query[dim], std_query[dim] = self.gp[dim].predict(z_query[dim], std=True)


        xyz = ['x','y','z']
        #plt.style.use('seaborn')
        sns.set_theme()
        plt.figure(figsize=(10, 6), dpi=100)

        for col in range(len(self.gp)):
            #print(np.ravel([f_grads[col](z_query[:,col])[d,d].full() for d in range(z_query.shape[0])]))
            plt.subplot(1,3,col+1)
            plt.plot(z_query[col], y_query[col])
            if z_train is not None and y_train is not None:
                plt.scatter(z_train[:,col], y_train[:,col], marker='+', c='k')
            plt.xlabel(f'Velocity {xyz[col]} [ms-1]')
            plt.ylabel(f'Drag acceleration {xyz[col]} [ms-2]')
            plt.legend(('m(z) interpolation', "m(z') training"))
            plt.fill_between(z_query[col], y_query[col] - 2*std_query[col], y_query[col] + 2*std_query[col], color='gray', alpha=0.2)
        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath, format="pdf", bbox_inches="tight")
        if show:
            plt.show()


    def __str__(self):
        return '\n\r'.join([f'GP{i}: ' + self.gp[i].__str__() for i in range(len(self.gp))])