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




    Nsim = 100 # number of simulation steps

    simulation_dt = 5e-4

    # initial condition

    quad = Quadrotor3D(payload=False, drag=True) # Controlled plant 
    quad_opt = quad_optimizer(quad, t_horizon=1, n_nodes=20, gpe=gpe) # computing optimal control over model of plant

    '''
    x_trajectory = quad_opt.square_trajectory(quad_opt.n_nodes, quad_opt.optimization_dt) # arbitrary trajectory
    u_trajectory = np.ones((x_trajectory.shape[0], 4))*0.16 # 0.16 is hover thrust 

    # set the created trajectory to the ocp solver
    yref, yref_N = quad_opt.set_reference_trajectory(x_trajectory, u_trajectory)
    '''

    x = np.array([0,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])

    x_optim = np.empty((Nsim+1, x.shape[0])) * np.NaN
    u_optim = np.empty((Nsim, 4)) * np.NaN


    x_target = np.array([10,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])

    yref, yref_N = quad_opt.set_reference_state(x_target)

    x_sim = x.reshape((1, x.shape[0]))
    x_pred_sim = np.empty((1, x.shape[0]))*np.NaN
    aero_drag_sim = np.empty((1, 3))*np.NaN
    GPE_pred_sim = np.empty((1, 3))*np.NaN
    u_sim = np.empty((1,4))*np.NaN
    yref_sim = np.empty((1, yref.shape[1]))*np.NaN

    # Set quad to start position
    quad.set_state(x)

    # IDEA : create a 3D array of Nsim, stateidx, n_node ## How to visualize?
    for i in tqdm(range(Nsim)):


        if i == Nsim/2:
            x_target = np.array([0,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])

        yref, yref_N = quad_opt.set_reference_state(x_target)
        yref_now = yref[0,:]

        # I dont think I need to run optimization more times as with the case of new opt
        x_opt_acados, w_opt_acados = quad_opt.run_optimization(x)

        u = w_opt_acados[0,:] # control to be applied to quad
        u_optim[i,:] = u
        x_optim[i,:] = x

        control_time = 0
        while control_time < quad_opt.optimization_dt: 
            # Uses the optimization model to predict one step ahead, used for gp fitting
            x_pred = quad_opt.discrete_dynamics(x, u, simulation_dt, body_frame=True)




            # Control the quad with the most recent u for the whole control period (multiple simulation steps for one optimization)
            quad.update(u, simulation_dt)
            x = np.array(quad.get_state(quaternion=True, stacked=True)) # state at the next optim step



            # x but in body frame referential
            x_to_save = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=True))

            # Save model aerodrag for GP validation, useful only when payload=False
            x_body = quad.get_state(quaternion=True, stacked=False, body_frame=False) # in world frame because get_aero_drag takes world frame velocity
            a_drag_body = quad.get_aero_drag(x_body, body_frame=True)
            
            # Add current state to array for dataset creation and visualisation
            u_sim = np.append(u_sim, u.reshape((1, u.shape[0])), axis=0)

            x_sim = np.append(x_sim, x_to_save.reshape((1, x_to_save.shape[0])), axis=0)
            x_pred_sim = np.append(x_pred_sim, x_pred.reshape((1,x.shape[0])), axis=0)
            yref_sim = np.append(yref_sim, yref_now.reshape((1, yref_now.shape[0])), axis=0)
            aero_drag_sim = np.append(aero_drag_sim, a_drag_body.reshape((1, a_drag_body.shape[0])), axis=0)

            # GPE prediction
            v_body = x_to_save[7:10]

            a_error_gpe = gpe.predict(v_body)
            #a_error_gpe = v_dot_q(a_error_gpe.T, x[3:7])
            GPE_pred_sim = np.append(GPE_pred_sim, a_error_gpe.reshape((1, -1)), axis=0)
            '''
            print("GPE prediction:")
            print(a_error_gpe)
            print("Measured drag:")
            print(a_drag_body)
            '''


            control_time += simulation_dt


    file = open('mpc_log.p', 'wb')
    pickle.dump([x_sim, u_sim, quad], file)
    file.close()

    save_trajectories_as_dict(x_sim, u_sim, x_pred_sim, aero_drag_sim, simulation_dt)



    fig = plt.figure()
    plt.subplot(131)
    plt.plot(x_sim[:,7], aero_drag_sim[:,0], 'r')
    plt.scatter(x_sim[:,7], GPE_pred_sim[:,0], c='c', s=1)
    plt.title('GP vs aero drag x')
    plt.legend(('Aero drag','GPE prediction'))
    plt.subplot(132)
    plt.plot(x_sim[:,8], aero_drag_sim[:,1], 'g')
    plt.scatter(x_sim[:,8], GPE_pred_sim[:,1], c='c', s=1)
    plt.title('GP vs aero drag y')
    plt.legend(('Aero drag','GPE prediction'))
    plt.subplot(133)
    plt.plot(x_sim[:,9], aero_drag_sim[:,2], 'b')  
    plt.scatter(x_sim[:,9], GPE_pred_sim[:,2], c='c', s=1)
    plt.title('GP vs aero drag z')
    plt.legend(('Aero drag','GPE prediction'))

    fig = plt.figure()
    plt.subplot(221)
    plt.plot(x_sim[:,0])
    plt.plot(x_sim[:,1])
    plt.plot(x_sim[:,2])
    plt.plot(x_pred_sim[:,0],'-.')
    plt.plot(x_pred_sim[:,1],'-.')
    plt.plot(x_pred_sim[:,2],'-.')
    plt.title('Position xyz')

    plt.subplot(222)
    plt.plot(x_sim[:,3])
    plt.plot(x_sim[:,4])
    plt.plot(x_sim[:,5])
    plt.plot(x_sim[:,6])
    plt.plot(x_pred_sim[:,3],'-.')
    plt.plot(x_pred_sim[:,4],'-.')
    plt.plot(x_pred_sim[:,5],'-.')
    plt.plot(x_pred_sim[:,6],'-.')
    plt.title('Quaternions')

    plt.subplot(223)
    plt.plot(x_sim[:,7])
    plt.plot(x_sim[:,8])
    plt.plot(x_sim[:,9])
    plt.plot(x_pred_sim[:,7],'-.')
    plt.plot(x_pred_sim[:,8],'-.')
    plt.plot(x_pred_sim[:,9],'-.')
    plt.title('Velocity xyz')

    plt.subplot(224)
    plt.plot(x_sim[:,10])
    plt.plot(x_sim[:,11])
    plt.plot(x_sim[:,12])
    plt.plot(x_pred_sim[:,10],'-.')
    plt.plot(x_pred_sim[:,11],'-.')
    plt.plot(x_pred_sim[:,12],'-.')
    plt.title('Angle rate')

    fig = plt.figure()
    plt.subplot(221)
    plt.plot(x_sim[:,0],'r')
    plt.plot(x_sim[:,1],'g')
    plt.plot(x_sim[:,2],'b')

    plt.plot(x_pred_sim[:,0],'r-.')
    plt.plot(x_pred_sim[:,1],'g-.')
    plt.plot(x_pred_sim[:,2],'b-.')

    plt.plot(yref_sim[:,0],'r--')
    plt.plot(yref_sim[:,1],'g--')
    plt.plot(yref_sim[:,2],'b--')
    plt.title('Position xyz')


    plt.subplot(222)
    plt.plot(u_sim[:,0],'r')
    plt.plot(u_sim[:,1],'g')
    plt.plot(u_sim[:,2],'b--')
    plt.plot(u_sim[:,3],'c--')
    plt.ylim([0,1.2])
    plt.title('Control u')

    #fig = plt.figure()
    plt.subplot(223)
    plt.plot(x_sim[:,3],'r')
    plt.plot(x_sim[:,4],'g')
    plt.plot(x_sim[:,5],'b')
    plt.plot(x_sim[:,6],'b')
    plt.title('Quaternion q')

    plt.subplot(224)
    plt.plot(aero_drag_sim[:,0],'b')
    plt.plot(x_pred_sim[:,8],'r-.')
    #plt.plot(aero_drag_sim[:,1],'g')
    #plt.plot(aero_drag_sim[:,2],'b')
    plt.plot(x_sim[:,8]**2*np.sign(x_sim[:,8]),'g--')
    #plt.plot(x_sim[:,9],'g--')
    #plt.plot(x_sim[:,10],'b--')
    plt.show()

def plot_intermediate(x_opt_acados, w_opt_acados):
    fig = plt.figure()
    plt.subplot(121)
    plt.plot(x_opt_acados[:,0],'r')
    plt.plot(x_opt_acados[:,1],'g')
    plt.plot(x_opt_acados[:,2],'b')

    #plt.plot(np.concatenate((yref[:quad_opt.n_nodes,0], [yref_N[0]])),'r--')
    #plt.plot(np.concatenate((yref[:quad_opt.n_nodes,1], [yref_N[1]])),'g--')
    #plt.plot(np.concatenate((yref[:quad_opt.n_nodes,2], [yref_N[2]])), 'b--')
    plt.title('Position xyz')
    
    plt.subplot(122)
    plt.plot(w_opt_acados[:,0],'r')
    plt.plot(w_opt_acados[:,1],'g')
    plt.plot(w_opt_acados[:,2],'b--')
    plt.plot(w_opt_acados[:,3],'c-.')
    plt.title('Control u')
    
    
if __name__ == '__main__':
    main()