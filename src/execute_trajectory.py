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
 
from cProfile import label
from math import ceil
import sys
import numpy as np
import casadi as cs
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quad import Quadrotor3D
#from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse

from utils import utils
from quad_opt import quad_optimizer
from utils.save_dataset import *
from trajectory_generation.generate_trajectory import generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

from trajectory_generation.TrajectoryGenerator import TrajectoryGenerator
import pickle
    
import argparse
import time

from gp.GP import *
from gp.GPE import GPEnsemble

from Logger import Logger
from Explorer import Explorer

def main():

 
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument

    dir_path = os.path.dirname(os.path.realpath(__file__))
    simulation_result_fname = os.path.join(dir_path, '..', 'outputs/python_simulation/data/executed_trajectory.pkl')
    simulation_plot_fname = os.path.join(dir_path, '..', 'outputs/python_simulation/img/executed_trajectory.pdf')

    parser.add_argument("-o", "--output", type=str, required=False, default=simulation_result_fname, help="Output data file")
    parser.add_argument("-p", "--plot_output", type=str, required=False, default=simulation_plot_fname, help="Output plot file")
    parser.add_argument("--gpe", type=int, required=True, help="Use trained GPE")
    parser.add_argument("--trajectory", type=int, required=True, help = "Trajectory type to use : 0 - From file, 1 - Random Waypoints, 2 - Circle")

    parser.add_argument("--v_max", type=float, required=True, help="Maximum velocity over trajectory") 
    parser.add_argument("--a_max", type=float, required=True, help="Maximum acceleration over trajectory")
    parser.add_argument("--show", type=int, required=False, default=1, help="plt.show() at the end of the script")
    # Read arguments from command line
    args = parser.parse_args()
    
        
    # TODO: Implement testing with different air resistance cooefficients/functions together with training GPes

    if args.gpe == 0:
        print("Not using GPE")
        gpe = None
    elif args.gpe == 1:
        ensemble_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'python_simulation', 'gp_models')
        gpe = GPEnsemble.fromdir(ensemble_path, "GP")
    elif args.gpe == 2:
        ensemble_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'python_simulation', 'gp_models')
        gpe = GPEnsemble.fromdir(ensemble_path, "RGP")
    else:
        raise ValueError("Invalid GPE argument")
        
    save_filepath = args.output
    logger = Logger(save_filepath)
    trajectory_generator = TrajectoryGenerator()

    v_max_limit = 30
    a_max_limit = 30

    # This musnt be faster than the quad is capable of
    # Max velocity and acceleration along the trajectory
    v_max = args.v_max
    a_max = args.a_max

    #explorer = Explorer(gpe)
    #v_max = explorer.velocity_to_explore

    if v_max > v_max_limit:
        v_max = v_max_limit
        print("v_max limited to " + str(v_max_limit))
    if a_max > a_max_limit:
        a_max = a_max_limit
        print("a_max limited to " + str(a_max_limit))


    simulation_dt = 5e-3 # Timestep simulation for the physics
    # 5e-4 is a good value for the acados dt

    # MPC prediction horizon
    t_lookahead = 1 # Prediction horizon duration
    n_nodes = 30 # Prediction horizon number of timesteps in t_lookahead


    # initial condition
    quad = Quadrotor3D(payload=False, drag=True) # Controlled plant 
    quad_opt = quad_optimizer(quad, t_horizon=t_lookahead, n_nodes=n_nodes, gpe=gpe) # computing optimal control over model of plant
    quad_nominal = quad_optimizer(quad, t_horizon=t_lookahead, n_nodes=n_nodes, gpe=None) # Predicting the state of the plant with control but without GP


    if args.trajectory == 0:
        # static trajectory
        trajectory_generator.sample_trajectory('static', v_max, a_max, quad_opt.optimization_dt)
        
    if args.trajectory == 1:
        # Generate trajectory as reference for the quadrotor
        # new trajectory
        hsize = 10
        num_waypoints = 4
        trajectory_generator.generate_random_waypoints(hsize, num_waypoints)
        trajectory_generator.sample_trajectory('random', v_max, a_max, quad_opt.optimization_dt)

    if args.trajectory == 2:
        # Circle trajectory
        radius = 50
        t_max = 30

        trajectory_generator.sample_circle_trajectory_accelerating(radius, v_max, t_max, quad_opt.optimization_dt)
        

    x_trajectory, t_trajectory = trajectory_generator.load_trajectory()
    
    #t_trajectory = t_trajectory[:len(t_trajectory)//4] # TODO: Remove this
    simulation_length = max(t_trajectory) # Simulation duration for this script
    # Simulation runs for simulation_length seconds and MPC is calculated every quad_opt.optimization_dt
    Nopt = round(simulation_length/quad_opt.optimization_dt) # number of times MPC control is calculated steps

    # initial condition
    x0 = np.array([0,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])
    

    simulate_trajectory(quad, quad_opt, quad_nominal, x0, x_trajectory, simulation_length, Nopt, simulation_dt, logger)
    


def simulate_trajectory(quad, quad_opt, quad_nominal, x0, x_trajectory, simulation_length, Nopt, simulation_dt, logger):

    x = x0
    # Ground truth data
    x_sim = list() # World reference frame
    yref_sim = list()
    u_sim = list()
    x_sim_body = list()# Body reference frame
    x_optim = list() # MPC prediction
    u_optim = list() # MPC prediction
    x_pred_sim = list()
    aero_drag_sim = list()
    GPE_pred_sim = list() 

    # Odometry is simulated every MPC timestep
    x_odom = list()
    x_ref_odom = list()
    w_odom = list()
    t_odom = list()
    t_odom = list()
    x_pred_odom = list()

    solution_times = list()
    cost_solutions = list()

    rgp_params = list()
    rgp_basis_vectors = list()
    mu_regress = list()
    a_dif = list()
    a_drag_list = list()
    v_body_list = list()

    # Set quad to start position
    #quad.set_state(x)
    # TODO: DO I NEED TO SET THE STATE OF THE QUAD HERE?

    print(f'Duration of simulation={simulation_length}, Number of simulation steps={Nopt}')
    simulation_time = 0
    for i in tqdm(range(Nopt)):

        # ----------- MPC control ----------------
        # Set the part of trajectory relevant for current time as the MPC reference
        x_ref = utils.get_reference_chunk(x_trajectory, i, quad_opt.n_nodes)
        yref, yref_N = quad_opt.set_reference_trajectory(x_ref)

        x = quad.get_state(quaternion=True, stacked=True) # Get current state of quad
        
        # I dont think I need to run optimization more times as with the case of new opt
        # TODO: Figure out why OCP gives different solutions both times it is run. warm start?
        time_before_mpc = time.time()
        x_opt_acados, w_opt_acados, t_cpu, cost_solution = quad_opt.run_optimization(x)
        elapsed_during_mpc = time.time() - time_before_mpc
        w = w_opt_acados[0,:].ravel() # control to be applied to quad

        x_pred = quad_nominal.discrete_dynamics(x, w, quad_opt.optimization_dt) # Predict next state of quad using optimal control

        #print(f'x_pred={x_pred}')
        #print(f'x_pred2={x_pred2}')


        # Save nlp solution diagnostics
        solution_times.append(t_cpu)
        cost_solutions.append(cost_solution)
        # Odometry every MPC timestep (100ms)    
        x_odom.append(x)
        x_ref_odom.append(yref[0,:13])
        w_odom.append(w)
        t_odom.append(simulation_time)
        x_pred_odom.append(x_pred)


        # ----------- Simulate the quadrotor ----------------
        control_time = 0
        # Simulate the quad plant with the optimal control until the next MPC optimization step is reached
        while control_time < quad_opt.optimization_dt: 
            # ----------- Simulate ----------------
            # Uses the optimization model to predict one step ahead, used for gp fitting
            #x_pred = quad_opt.discrete_dynamics(x, w, simulation_dt, body_frame=True)
            # Control the quad with the most recent u for the whole control period (multiple simulation steps for one optimization)
            quad.update(w, simulation_dt)

            '''
            # ----------- Save simulation results ----------------
            x = np.array(quad.get_state(quaternion=True, stacked=True)) # state at the next optim step

            # Save model aerodrag for GP validation, useful only when payload=False
            x_body_for_drag = quad.get_state(quaternion=True, stacked=False, body_frame=False) # in world frame because get_aero_drag takes world frame velocity
            a_drag_body = quad.get_aero_drag(x_body_for_drag, body_frame=True)
            
            # Save simulation results
            x_world = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=False)) # World frame referential
            x_body = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=True)) # Body frame referential

            '''

            '''
            # Save simulation results
            # Add current simulation results to list for dataset creation and visualisation
            x_sim.append(x_world)
            u_sim.append(w)
            x_sim_body.append(x_body)
            
            x_pred_sim.append(x_pred)
            yref_now = yref[0,:]
            yref_sim.append(yref_now)
            aero_drag_sim.append(a_drag_body)
            '''




            # Counts until the next MPC optimization step is reached
            control_time += simulation_dt
        

        # ----------------- Regress RGP -----------------
        if quad_opt.gpe is not None:
            if quad_opt.gpe.type == 'RGP':
                rgp_basis_vectors = [quad_opt.gpe.gp[d].X
                        for d in range(len(quad_opt.gpe.gp))]
                if logger.dictionary:
                    x_pred_minus_1 = logger.dictionary['x_pred_odom'][-1]
                else:
                    x_pred_minus_1 = x

                rgp_mu_g_t, rgp_C_g_t = quad_opt.regress_and_update_RGP_model(x, x_pred_minus_1)
                rgp_theta = quad_opt.gpe.get_theta()
            else:
                # If not using RGP, set these to None for logging
                rgp_basis_vectors = None
                rgp_mu_g_t = None
                rgp_C_g_t = None
                rgp_theta = None
        
        # ------- Log data -------
        if logger is not None:
            dict_to_log = {"x_odom": x, "x_pred_odom": x_pred, "x_ref": x_ref[0,:], "t_odom": simulation_time, \
                "w_odom": w, 't_cpu': t_cpu, "cost_solution": cost_solution, \
                    "rgp_basis_vectors" : rgp_basis_vectors, "rgp_mu_g_t": rgp_mu_g_t, "rgp_C_g_t": rgp_C_g_t, "rgp_theta": rgp_theta}
            
            logger.log(dict_to_log)
        # Counts until simulation is finished
        simulation_time += quad_opt.optimization_dt
    

    '''
    
    
    t = np.linspace(0, simulation_length, len(x_sim))

    # Convert lists to numpy arrays
    x_sim = np.squeeze(np.array(x_sim))
    x_sim_body = np.squeeze(np.array(x_sim_body))
    u_sim = np.squeeze(np.array(u_sim))
    aero_drag_sim = np.squeeze(np.array(aero_drag_sim))
    x_pred_sim = np.squeeze(np.array(x_pred_sim))

    x_odom = np.squeeze(np.array(x_odom))
    x_ref_odom = np.squeeze(np.array(x_ref_odom))  
    w_odom = np.squeeze(np.array(w_odom))
    solution_times = np.squeeze(np.array(solution_times))
    cost_solutions = np.squeeze(np.array(cost_solutions))
    t_odom = np.squeeze(np.array(t_odom))
    x_pred_odom = np.squeeze(np.array(x_pred_odom))


    data = dict()

    rmse_pos = np.sqrt(np.mean((yref_now[:3] - x[:3])**2))
    #rmse_pos = np.append(rmse_pos, rmse_pos_now)
    data['x_sim'] = x_sim
    data['gpe'] = quad_opt.gpe is not None
    data['rmse_pos'] = rmse_pos
    data['u'] = u_sim
    data['aero_drag'] = aero_drag_sim
    data['x_pred_sim'] = x_pred_sim
    # need the dt to calculate a_error
    data['dt'] = simulation_dt
    data['t'] = t

    # ----- These are identical to the gazebo dataset logger -----
    data['x_odom'] = x_odom
    data['x_ref'] = x_ref_odom
    data['w_odom'] = w_odom
    data['t_cpu'] = solution_times
    data['cost_solution'] = cost_solutions
    data['t_odom'] = t_odom
    data['x_pred_odom'] = x_pred_odom

    if quad_opt.gpe is not None:
        if quad_opt.gpe.type == 'RGP':
            data['rgp_basis_vectors'] = rgp_basis_vectors
            data['rgp_params'] = rgp_params

    '''
    
    #save_dict(data, logger.filepath_dict)
    logger.save_log()
    #print(f'Saved simulated data to {save_filepath}')


    '''
    
    if quad_opt.gpe is not None:
        if quad_opt.gpe.type == "RGP":
            print('Rendering animation...')
            
            X_ = [None]*len(quad_opt.gpe.gp)
            for d in range(len(quad_opt.gpe.gp)):
                X_[d] = quad_opt.gpe.gp[d].X
            
            def animate(i):
                
                for d in range(len(quad_opt.gpe.gp)):
                    scat_basis_vectors[d].set_offsets(np.array([X_[d].ravel(), mu_regress[i][d].ravel()]).T)
                    pbar.update()

            animation.writer = animation.writers['ffmpeg']
            plt.ioff() # Turn off interactive mode to hide rendering animations

            plt.style.use('fast')
            sns.set_style("whitegrid")

            #gs = gridspec.GridSpec(2, 2)
            scat_basis_vectors = [None]*len(quad_opt.gpe.gp)
            fig = plt.figure(figsize=(10,10), dpi=100)
            for d in range(len(quad_opt.gpe.gp)): 
                ax = fig.add_subplot(1,3,d+1)
                scat_basis_vectors[d] = ax.scatter([], [], marker='o', label='Basis Vectors')
                ax.set_xlim(( min(X_[d])) , max(X_[d]))
                ax.set_ylim(( min([min(mu_regress[i][d]) for i in range(len(mu_regress))])) , max([max(mu_regress[i][d]) for i in range(len(mu_regress))]))
                #ax.set_ylim((min((min(y_), min(y_t), min(y_true))) , max((max(y_), max(y_t), max(y_true)))))
                #ax.set_ylim((-10,10))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Recursive Gaussian Process')
                ax.legend()


            dir_path = os.path.dirname(os.path.realpath(__file__))
            save_anim_path = os.path.join(dir_path, '..', 'outputs/graphics/regression_during_python_sim')
            pbar = tqdm(total=len(mu_regress))
            ani = animation.FuncAnimation(fig, animate, frames=len(mu_regress), interval=500)     
            ani.save(save_anim_path + '.mp4', writer='ffmpeg', fps=10, dpi=100)
            ani.save(save_anim_path + '.gif', writer='imagemagick', fps=10, dpi=100)
            pbar.close()
            #plt.show()


    '''



    
if __name__ == '__main__':
    main()