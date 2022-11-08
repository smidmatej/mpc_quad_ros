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

from quad import Quadrotor3D
from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse
from utils import utils
from quad_opt import quad_optimizer
from utils.save_dataset import *
from trajectory_generation.generate_trajectory import generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

import pickle
    
import argparse
    
from gp.gp import *
from gp.gp_ensemble import GPEnsemble




#token = '5793137870:AAGN2y8dLZezvlf__O5yMvKlIlEDO9hTqJI'
#schat_id = '5512359229'
def main():

 
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument

    parser.add_argument("-o", "--output", type=str, required=False, default='data/simulated_flight.pkl', help="Output filename")
    parser.add_argument("--gpe", type=int, required=True, help="Use trained GPE")
    parser.add_argument("--trajectory", type=int, required=True, help = "Trajectory type to use : 0 - From file, 1 - Random Waypoints, 2 - Circle")

    parser.add_argument("--v_max", type=float, required=True, help="Maximum velocity over trajectory") 
    parser.add_argument("--a_max", type=float, required=True, help="Maximum acceleration over trajectory")
    parser.add_argument("--show", type=int, required=False, default=1, help="plt.show() at the end of the script")
    parser.add_argument("-pf", "--plot_filename", type=str, required=False, default="img/trajectory_tracking.pdf", help="Save filename for plot")

    
    # Read arguments from command line
    args = parser.parse_args()
    
    
    # TODO: Implement testing with different air resistance cooefficients/functions together with training GPes

    if args.gpe:
        ensemble_path = "gp/models/ensemble"
        gpe = GPEnsemble(3)
        gpe.load(ensemble_path)
    else:
        gpe = None





    # This musnt be faster than the quad is capable of
    # Max velocity and acceleration along the trajectory
    v_max = args.v_max
    a_max = args.a_max



    output_trajectory_filename = 'trajectory_generation/trajectories/trajectory_sampled.csv'


 
    simulation_dt = 5e-3 # Timestep simulation for the physics
    # 5e-4 is a good value for the acados dt

    # MPC prediction horizon
    t_lookahead = 1 # Prediction horizon duration
    n_nodes = 50 # Prediction horizon number of timesteps in t_lookahead


    # initial condition
    quad = Quadrotor3D(payload=False, drag=True) # Controlled plant 
    quad_opt = quad_optimizer(quad, t_horizon=t_lookahead, n_nodes=n_nodes, gpe=gpe) # computing optimal control over model of plant
    


    if args.trajectory == 0:
        # static trajectory
        waypoint_filename = 'trajectory_generation/waypoints/static_waypoints.csv'
        # Create trajectory from waypoints with the same dt as the MPC control frequency    
        create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, quad_opt.optimization_dt)
        # trajectory has a specific time step that I do not respect here
        x_trajectory, t_trajectory = utils.load_trajectory(output_trajectory_filename)
    


    if args.trajectory == 1:
        # Generate trajectory as reference for the quadrotor
        # new trajectory
        hsize = 100 # halfsize of the cube in which to generate the waypoints
        num_waypoints = 10
        waypoint_filename = 'trajectory_generation/waypoints/random_waypoints.csv'
        generate_random_waypoints(waypoint_filename, hsize=hsize, num_waypoints=num_waypoints)
        create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, quad_opt.optimization_dt)

        # trajectory has a specific time step that I do not respect here
        x_trajectory, t_trajectory = utils.load_trajectory(output_trajectory_filename)

    if args.trajectory == 2:
        # Circle trajectory
        radius = 50
        t_max = 30

        circle_trajectory_filename = 'trajectory_generation/trajectories/circle_trajectory.csv'
        generate_circle_trajectory_accelerating(circle_trajectory_filename, radius, v_max, t_max=t_max, dt=quad_opt.optimization_dt)
        # trajectory has a specific time step that I do not respect here
        x_trajectory, t_trajectory = utils.load_trajectory(circle_trajectory_filename)




    t_simulation = max(t_trajectory) # Simulation duration for this script

    # Simulation runs for t_simulation seconds and MPC is calculated every quad_opt.optimization_dt
    Nopt = round(t_simulation/quad_opt.optimization_dt) # number of times MPC control is calculated steps
    Nsim = round(t_simulation/simulation_dt)
    
    #Nopt = round(t_simulation/(t_trajectory[1] - t_trajectory[0]))
    u_trajectory = np.ones((x_trajectory.shape[0], 4))*0.16 # 0.16 is hover thrust 

    # set the created trajectory to the ocp solver
    traj_dt = t_trajectory[1] - t_trajectory[0]

    #undersampling = round(quad_opt.optimization_dt/(traj_dt))
    #undersampling = 1
    yref, yref_N = quad_opt.set_reference_trajectory(x_trajectory, u_trajectory)

    # initial condition
    x = np.array([0,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])



    # Allocate memory for simulation results saving
    solution_times = list()
    cost_solutions = list()

    x_sim = x.reshape((1, x.shape[0]))

    x_optim = np.empty((Nopt+1, x.shape[0])) * np.NaN
    u_optim = np.empty((Nopt, 4)) * np.NaN
    x_pred_sim = np.empty((1, x.shape[0]))*np.NaN
    aero_drag_sim = np.empty((1, 3))*np.NaN
    GPE_pred_sim = np.empty((1, 3))*np.NaN
    u_sim = np.empty((1,4))*np.NaN
    x_sim_body = np.zeros((1, x.shape[0]))
    yref_sim = np.empty((1, yref.shape[1]))*np.NaN

    rmse_pos = np.zeros((1,))*0

    # Set quad to start position
    quad.set_state(x)

    print(f'Duration of simulation={t_simulation}, Number of simulation steps={Nopt}')
    # IDEA : create a 3D array of Nopt, stateidx, n_node ## How to visualize?
    simulation_time = 0
    for i in tqdm(range(Nopt)):
        #print(f'rmse_pos={rmse_pos[-1]}')
        # Set the part of trajectory relevant for current time as the MPC reference
        x_ref = utils.get_reference_chunk(x_trajectory, i, quad_opt.n_nodes)
        yref, yref_N = quad_opt.set_reference_trajectory(x_ref)


        # I dont think I need to run optimization more times as with the case of new opt
        # TODO: Figure out why OCP gives different solutions both times it is run. warm start?
        x_opt_acados, w_opt_acados, t_cpu, cost_solution = quad_opt.run_optimization(x)

        # Save nlp solution diagnostics
        solution_times.append(t_cpu)
        cost_solutions.append(cost_solution)

        u = w_opt_acados[0,:] # control to be applied to quad

        # Save optimal control and state
        u_optim[i,:] = u
        x_optim[i,:] = x
                    

        control_time = 0
        # Simulate the quad plant with the optimal control until the next MPC optimization step is reached
        while control_time < quad_opt.optimization_dt: 
            # Uses the optimization model to predict one step ahead, used for gp fitting
            x_pred = quad_opt.discrete_dynamics(x, u, simulation_dt, body_frame=True)

            # Control the quad with the most recent u for the whole control period (multiple simulation steps for one optimization)
            quad.update(u, simulation_dt)
            x = np.array(quad.get_state(quaternion=True, stacked=True)) # state at the next optim step



            # Save model aerodrag for GP validation, useful only when payload=False
            x_body_for_drag = quad.get_state(quaternion=True, stacked=False, body_frame=False) # in world frame because get_aero_drag takes world frame velocity
            a_drag_body = quad.get_aero_drag(x_body_for_drag, body_frame=True)
            
            

            x_world = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=False)) # World frame referential
            x_body = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=True)) # Body frame referential

            # Save simulation results
            # Add current state to array for dataset creation and visualisation
            u_sim = np.append(u_sim, u.reshape((1, u.shape[0])), axis=0)

            x_sim = np.append(x_sim, x_world.reshape((1, x_world.shape[0])), axis=0)
            x_sim_body = np.append(x_sim_body, x_body.reshape((1, x_body.shape[0])), axis=0)
            
            x_pred_sim = np.append(x_pred_sim, x_pred.reshape((1,x.shape[0])), axis=0)
            yref_now = yref[0,:]
            yref_sim = np.append(yref_sim, yref_now.reshape((1, yref_now.shape[0])), axis=0)
            aero_drag_sim = np.append(aero_drag_sim, a_drag_body.reshape((1, a_drag_body.shape[0])), axis=0)

            #print(f' x={x}')
            #print(f'yref_now={yref_now}')
            rmse_pos_now = np.sqrt(np.mean((yref_now[:3] - x[:3])**2))
            #print(f'rmse_pos_now={rmse_pos_now}')
            #break
            

            rmse_pos = np.append(rmse_pos, rmse_pos_now)
            # Counts until the next MPC optimization step is reached
            control_time += simulation_dt
        # Counts until simulation is finished
        simulation_time += quad_opt.optimization_dt
    

    t = np.linspace(0, t_simulation, x_sim.shape[0])

    # Old way of saving data
    #file = open('mpc_log.p', 'wb')
    #pickle.dump([x_sim, u_sim, quad], file)
    #file.close()

    # New way of saving data 
    #save_trajectories_as_dict(x_sim, u_sim, x_pred_sim, aero_drag_sim, t, simulation_dt, 'data/simulated_flight.pkl')

    data = dict()

    # measured state
    data['p'] = x_sim[:,0:3]
    data['q'] = x_sim[:,3:7]
    data['v'] = x_sim[:,7:10]
    data['w'] = x_sim[:,10:13]
    # body frame velocity
    data['v_body'] = x_sim_body[:,7:10]

    data['gpe'] = args.gpe
    data['rmse_pos'] = rmse_pos

    data['u'] = u_sim
    data['aero_drag'] = aero_drag_sim

    # predicted state
    data['p_pred'] = x_pred_sim[:,0:3]
    data['q_pred'] = x_pred_sim[:,3:7]
    data['v_pred'] = x_pred_sim[:,7:10]
    data['w_pred'] = x_pred_sim[:,10:13]

    # need the dt to calculate a_error
    data['dt'] = simulation_dt
    data['t'] = t

    
    save_dict(data, args.output)
    print(f'Saved simulated data to {args.output}')

    sns.set_style("whitegrid")

    '''
    plt.figure()
    plt.plot(x_sim_body[:,7], aero_drag_sim[:,0], 'r', label='aero_drag_x')
    plt.plot(x_sim_body[:,8], aero_drag_sim[:,1], 'g', label='aero_drag_y')
    plt.plot(x_sim_body[:,9], aero_drag_sim[:,2], 'b', label='aero_drag_z')
    '''

    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.subplot(241)
    plt.plot(t, x_sim[:,0], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,1], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,2], 'b', linewidth=0.8)
    plt.plot(t, yref_sim[:,0], 'r--', linewidth=0.8)
    plt.plot(t, yref_sim[:,1], 'g--', linewidth=0.8)
    plt.plot(t, yref_sim[:,2], 'b--', linewidth=0.8)
    plt.title('Position xyz')


    plt.subplot(242)
    plt.plot(t, x_sim[:,3], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,4], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,5], 'b', linewidth=0.8)
    plt.plot(t, x_sim[:,6], 'c', linewidth=0.8)
    plt.plot(t, yref_sim[:,3], 'r--', linewidth=1)
    plt.plot(t, yref_sim[:,4], 'g--', linewidth=1)
    plt.plot(t, yref_sim[:,5], 'b--', linewidth=1)
    plt.plot(t, yref_sim[:,6], 'c--', linewidth=1)
    plt.title('Quaternion q')

    plt.subplot(243)
    plt.plot(t, x_sim[:,7], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,8], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,9], 'b', linewidth=0.8)
    plt.plot(t, np.linalg.norm(x_sim[:,7:10], axis=1), 'c', linewidth=0.8, label='vmax')
    plt.plot(t, -np.linalg.norm(x_sim[:,7:10], axis=1), 'c', linewidth=0.8, label='vmax')
    plt.plot(t, yref_sim[:,7], 'r--', linewidth=0.8)
    plt.plot(t, yref_sim[:,8], 'g--', linewidth=0.8)
    plt.plot(t, yref_sim[:,9], 'b--', linewidth=0.8)
    plt.plot(t, np.linalg.norm(yref_sim[:,7:10], axis=1), 'c--', linewidth=0.8, label='vmax_ref')
    plt.plot(t, -np.linalg.norm(yref_sim[:,7:10], axis=1), 'c--', linewidth=0.8, label='vmax_ref')
    plt.plot(t, np.repeat(v_max, repeats=len(t)), 'k--', linewidth=0.8, label='vmax')
    plt.plot(t, -np.repeat(v_max, repeats=len(t)), 'k--', linewidth=0.8, label='vmax')


    plt.title('Velocity xyz')

    plt.subplot(244)
    plt.plot(t, x_sim[:,10], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,11], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,12], 'b', linewidth=0.8)
    plt.plot(t, yref_sim[:,10], 'r--', linewidth=0.8)
    plt.plot(t, yref_sim[:,11], 'g--', linewidth=0.8)
    plt.plot(t, yref_sim[:,12], 'b--', linewidth=0.8)
    plt.title('Angle rate xyz')

    plt.subplot(245)
    plt.plot(t, u_sim[:,0], 'r', linewidth=0.8)
    plt.plot(t, u_sim[:,1], 'g', linewidth=0.8)
    plt.plot(t, u_sim[:,2], 'b', linewidth=0.8)
    plt.plot(t, u_sim[:,3], 'c', linewidth=0.8)
    plt.ylim([0,1.2])
    plt.title('Control u')

    plt.subplot(246)
    plt.plot(solution_times, linewidth=0.8)
    #plt.plot([quad_opt.optimization_dt]*len(solution_times))
    plt.title(f'Total optimization time: {np.sum(solution_times).round(2)}s')
    #plt.legend(('MPC solution time', 'quad_opt.optimization_dt'))
    

    plt.subplot(247)
    plt.plot(cost_solutions, linewidth=0.8)
    #plt.plot([quad_opt.optimization_dt]*len(solution_times))
    plt.title('Cost of solution')

    plt.subplot(248)
    plt.plot(t, aero_drag_sim[:,0], 'r', linewidth=0.8)
    plt.plot(t, aero_drag_sim[:,1], 'g', linewidth=0.8)
    plt.plot(t, aero_drag_sim[:,2], 'b', linewidth=0.8)
    #plt.plot(rmse_pos, linewidth=0.8)
    #plt.plot([quad_opt.optimization_dt]*len(solution_times))
    #plt.title('Position RMSE')
    #plt.legend(('MPC solution time', 'quad_opt.optimization_dt'))
    
    
    
    plt.tight_layout()

    
    plt.savefig(args.plot_filename, format="pdf", bbox_inches="tight")
    print(f'Saved generated figure to {args.plot_filename}')
    if args.show:
        plt.show()

    


    '''
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(t,x_sim[:,0])
    plt.plot(t,x_sim[:,1])
    plt.plot(t,x_sim[:,2])
    plt.plot(t,x_pred_sim[:,0],'-.')
    plt.plot(t,x_pred_sim[:,1],'-.')
    plt.plot(t,x_pred_sim[:,2],'-.')
    plt.title('Position xyz')


    plt.subplot(222)
    plt.plot(t,x_sim[:,3])
    plt.plot(t,x_sim[:,4])
    plt.plot(t,x_sim[:,5])
    plt.plot(t,x_sim[:,6])
    plt.plot(t,x_pred_sim[:,3],'-.')
    plt.plot(t,x_pred_sim[:,4],'-.')
    plt.plot(t,x_pred_sim[:,5],'-.')
    plt.plot(t,x_pred_sim[:,6],'-.')
    plt.title('Quaternions')

    plt.subplot(223)
    plt.plot(t,x_sim[:,7])
    plt.plot(t,x_sim[:,8])
    plt.plot(t,x_sim[:,9])
    plt.plot(t,x_pred_sim[:,7],'-.')
    plt.plot(t,x_pred_sim[:,8],'-.')
    plt.plot(t,x_pred_sim[:,9],'-.')
    plt.title('Velocity xyz')

    plt.subplot(224)
    plt.plot(t,x_sim[:,10])
    plt.plot(t,x_sim[:,11])
    plt.plot(t,x_sim[:,12])
    plt.plot(t,x_pred_sim[:,10],'-.')
    plt.plot(t,x_pred_sim[:,11],'-.')
    plt.plot(t,x_pred_sim[:,12],'-.')
    plt.title('Angle rate')
    plt.tight_layout()
    '''
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