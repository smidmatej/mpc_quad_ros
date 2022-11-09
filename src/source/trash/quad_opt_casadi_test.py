import numpy as np
import casadi as cs
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from tqdm import tqdm

from quad import Quadrotor3D
from utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse
from quad_opt import quad_optimizer
from save_dataset import save_trajectories_as_dict

import pickle
    
    
from gp.gp import *
from gp.gp_ensemble import GPEnsemble

def main():


    # load GPE 


    save_path = "gp/models/ensemble"
    gpe = GPEnsemble(3)
    gpe.load(save_path)




    Nsim = 50 # number of simulation steps

    simulation_dt = 5e-4
    x = np.array([0,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])
    u = np.ones((4,1))*0

    # initial condition

    quad = Quadrotor3D(payload=False, drag=True) # Controlled plant 
    quad_opt = quad_optimizer(quad, t_horizon=1, n_nodes=20, gpe=gpe) # computing optimal control over model of plant
    print(quad_opt.f_dict['f_augment'](x=x,u=u))
    print(quad_opt.f_dict['f_dyn'](x=x,u=u))

if __name__ == '__main__':
    main()