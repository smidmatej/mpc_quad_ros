import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from gp.gp import GPR
except ImportError:
    from gp import GPR
import numpy as np
import casadi as cs
import os
import time

class GPEnsemble:

    
    def __init__(self, number_of_dimensions=0):
        self.gp = [None]*number_of_dimensions
        self.number_of_dimensions = number_of_dimensions
        
    def add_gp(self, new_gp, dim):
        self.gp[dim] = new_gp
        
    def predict(self, z, std=False):
        ### TODO: Add std and variance to casadi prediction ###

        out_j = [None]*self.number_of_dimensions
        if isinstance(z, cs.MX):
            for n in range(len(self.gp)):
                #print(z.shape)
                z_in_dim = z[:,n]
                #if z_in_dim.shape == (1,)
                #print(z_in_dim)
                out_j[n] = self.gp[n].predict(z_in_dim)
                #print(type(out_j[n]))
            concat = [out_j[n] for n in range(len(out_j))]
            #print(len(concat))
            out = cs.horzcat(*concat)
            return out
        else:
            # in case of prediction on one sample
            z = np.atleast_2d(z)
            if std:
                # std requested, need to get std from all gps
                std = [None]*self.number_of_dimensions
                for n in range(len(self.gp)):
                    out_j[n], std[n] = self.gp[n].predict(z[:,n].reshape(-1,1), std=True)
                out = np.concatenate(out_j, axis=1)
                return out, std
            else:
                # Nobody wants std
                for n in range(len(self.gp)):
                    out_j[n] = self.gp[n].predict(z[:,n].reshape(-1,1))
                out = np.concatenate(out_j, axis=1)
                return out
            
        
    
    

    def fit(self):
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
        assert z.shape[1] == self.number_of_dimensions, f"z needs to be n x d,  z.shape={z.shape}, GPE.number_of_dimensions={self.number_of_dimensions}"

        f_jacobs = list()
        for col in range(self.number_of_dimensions):
            f_jacobs.append(self.gp[col].jacobian(z[:,col]))
        return f_jacobs

    def save(self, path, xyz=True):

        if os.path.exists(path):
            print("Folder already exists, saving models inside")
        else:
            os.makedirs(path)
            print("Folder created, saving models inside")

        if xyz: 
            xyz_name = ['model_x','model_y','model_z']
            # GPE contains 3 GPs, one for each dimension

            for gpr_index in range(len(self.gp)):
                path_with_name = os.path.join(path, xyz_name[gpr_index])
                self.gp[gpr_index].save(path_with_name)
        
        else:
            raise NotImplementedError

    def load(self, path, xyz=True):
        print("Loading GPEnsemble from path: ", path)
        if xyz: 
            xyz_name = ['model_x','model_y','model_z']
            # GPE contains 3 GPs, one for each dimension

            # Discard the old GPE contents
            #self.gp = list()
            for gpr_index in range(len(xyz_name)):
                path_with_name = os.path.join(path, xyz_name[gpr_index])
                # Create a new empty GPR and add it to GPE
                self.add_gp(GPR(None,None,None,None), gpr_index)
                
                # Call the load method of the new empty GPR
                self.gp[gpr_index].load(path_with_name)

            self.number_of_dimensions = len(self.gp)
        else:
            raise NotImplementedError

    def plot(self, z_train=None, y_train=None, filepath=None, show=True):

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