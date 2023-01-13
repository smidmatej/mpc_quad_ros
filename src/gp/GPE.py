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

    def __init__(self, number_of_dimensions : int = 0, type : str = 'GP') -> None:
        """
        Ensamble of (Recursive) Gaussian Process regressors. Holds a list of GP or RGP objects, one for each dimension of the output.
        :param number_of_dimensions: Number of dimensions of the output
        :param type: Type of GP to be used. 'GP' or 'RGP'
        """
        self.gp = [None]*number_of_dimensions
        self.number_of_dimensions = number_of_dimensions
        
        self.type = type

    def __init__(self, gps : list = []) -> None:
        """
        Ensamble of (Recursive) Gaussian Process regressors. Holds a list of GP or RGP objects, one for each dimension of the output.
        :param gps: List of GP or RGP objects
        """
        self.gp = gps
        self.number_of_dimensions = len(gps)
        
        if all([isinstance(g, GP) for g in self.gp]):
            self.type = 'GP'
        elif all([isinstance(g, RGP) for g in self.gp]):
            self.type = 'RGP'
        else:
            raise ValueError("All GP objects in the list must be of the same type")


        
    def predict(self, X_t : float, std=False):
        ### TODO: Add std and variance to casadi prediction ###

        out_j = [None]*self.number_of_dimensions
        if isinstance(X_t, cs.MX):
            # ----------------- Casadi prediction -----------------
            for n in range(len(self.gp)):
                out_j[n] = self.gp[n].predict(X_t[:,n])

            concat = [out_j[n] for n in range(len(out_j))]
            out = cs.horzcat(*concat)
            return out
        else:
            # ----------------- Numpy prediction -----------------
            # in case of prediction on one sample
            X_t = np.atleast_2d(X_t)
            if std:
                # std requested, need to get std from all gps
                std = [None]*self.number_of_dimensions
                for n in range(len(self.gp)):
                    out_j[n], std[n] = self.gp[n].predict(X_t[:,n].reshape(-1,1), std=True)
                out = np.concatenate(out_j, axis=1)
                return out, std
            else:
                # Nobody wants std
                for n in range(len(self.gp)):
                    out_j[n] = self.gp[n].predict(X_t[:,n].reshape(-1,1))
                out = np.concatenate(out_j, axis=1)
                return out
            
        
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

        assert z.shape[1] == self.number_of_dimensions, f"z needs to be n x d,  z.shape={z.shape}, GPE.number_of_dimensions={self.number_of_dimensions}"

        f_jacobs = list()
        for col in range(self.number_of_dimensions):
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
 
        if self.type == 'RGP':
            xyz_name = ['rgp_x','rgp_y','rgp_z']
            for i_gp in range(len(self.gp)):
                path_with_name = os.path.join(path, xyz_name[i_gp])
                RGP.save(self.gp[i_gp], path_with_name)

        elif self.type == 'GP':

            xyz_name = ['gp_x','gp_y','gp_z']
            # GPE contains 3 GPs, one for each dimension
            for i_gp in range(len(self.gp)):
                path_with_name = os.path.join(path, xyz_name[i_gp])
                GP.save(self.gp[i_gp], path_with_name)
        


    def load(self, path : str) -> None:
        """
        Loads GPs from path and adds them to the GPEnsemble
        :param: path: folder path to load the models. Models will be loaded from inside the folder with the name model_x, model_y, model_z
        """
        print("Loading GPEnsemble from path: ", path)

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


        self.number_of_dimensions = len(self.gp)


    def plot(self, z_train=None, y_train=None, filepath=None, show=True):

        # TODO: Why am I passing the z_train and y_train here? I can just use the data from the GPE
        z_query = np.concatenate([np.arange(-20,20,0.5).reshape(-1,1) for i in range(3)], axis=1)
        y_query, std_query = self.predict(z_query, std=True)

        xyz = ['x','y','z']
        #plt.style.use('seaborn')
        sns.set_theme()
        plt.figure(figsize=(10, 6), dpi=100)

        for col in range(y_query.shape[1]):
            #print(np.ravel([f_grads[col](z_query[:,col])[d,d].full() for d in range(z_query.shape[0])]))
            plt.subplot(1,3,col+1)
            plt.plot(z_query[:,col], y_query[:,col])
            if z_train is not None and y_train is not None:
                plt.scatter(z_train[:,col], y_train[:,col], marker='+', c='k')
            plt.xlabel(f'Velocity {xyz[col]} [ms-1]')
            plt.ylabel(f'Drag acceleration {xyz[col]} [ms-2]')
            plt.legend(('m(z) interpolation', "m(z') training"))
            plt.fill_between(z_query[:,col], y_query[:,col] - 2*std_query[col], y_query[:,col] + 2*std_query[col], color='gray', alpha=0.2)
        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath, format="pdf", bbox_inches="tight")
        if show:
            plt.show()


    def __str__(self):
        return '\n\r'.join([f'GP{i}: ' + self.gp[i].__str__() for i in range(len(self.gp))])