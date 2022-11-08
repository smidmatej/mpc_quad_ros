from gp import GPR
import numpy as np



class GPEnsemble:

    
    def __init__(self, number_of_dimensions):
        self.gp = [None]*number_of_dimensions
        self.number_of_dimensions = number_of_dimensions
        
    def add_gp(self, new_gp, dim):
        self.gp[dim] = new_gp
        
    def predict(self, z):
        
        out_j = [None]*self.number_of_dimensions
        for n in range(len(self.gp)):
            out_j[n] = self.gp[n].predict(z[:,n].reshape(-1,1)).reshape(-1,1)
        out = np.concatenate(out_j, axis=1)
        return out
    
    def fit(self):
        for gpr in self.gp:
            gpr.fit()