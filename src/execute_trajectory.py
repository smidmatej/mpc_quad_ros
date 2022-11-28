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
from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse
from utils import utils
from quad_opt import quad_optimizer
from utils.save_dataset import *
from trajectory_generation.generate_trajectory import generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

from trajectory_generation.TrajectoryGenerator import TrajectoryGenerator
import pickle
    
import argparse
    
from gp.gp import *
from gp.gp_ensemble import GPEnsemble

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

    if args.gpe:
        ensemble_path = "gp/models/ensemble"
        gpe = GPEnsemble(3)
        gpe.load(ensemble_path)
    else:
        gpe = None



    trajectory_generator = TrajectoryGenerator()

    # This musnt be faster than the quad is capable of
    # Max velocity and acceleration along the trajectory
    v_max = args.v_max
    a_max = args.a_max



    #output_trajectory_filename = 'trajectory_generation/trajectories/trajectory_sampled.csv'


 
    simulation_dt = 5e-3 # Timestep simulation for the physics
    # 5e-4 is a good value for the acados dt

    # MPC prediction horizon
    t_lookahead = 1 # Prediction horizon duration
    n_nodes = 30 # Prediction horizon number of timesteps in t_lookahead


    # initial condition
    quad = Quadrotor3D(payload=False, drag=True) # Controlled plant 
    quad_opt = quad_optimizer(quad, t_horizon=t_lookahead, n_nodes=n_nodes, gpe=gpe) # computing optimal control over model of plant
    


    if args.trajectory == 0:
        # static trajectory
        trajectory_generator.sample_trajectory('static', v_max, a_max, quad_opt.optimization_dt)
        
        '''
        waypoint_filename = 'trajectory_generation/waypoints/static_waypoints.csv'
        # Create trajectory from waypoints with the same dt as the MPC control frequency    
        create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, quad_opt.optimization_dt)
        # trajectory has a specific time step that I do not respect here
        x_trajectory, t_trajectory = utils.load_trajectory(output_trajectory_filename)
        '''

    


    if args.trajectory == 1:
        # Generate trajectory as reference for the quadrotor
        # new trajectory
        hsize = 10
        num_waypoints = 10
        trajectory_generator.generate_random_waypoints(hsize, num_waypoints)
        trajectory_generator.sample_trajectory('random', v_max, a_max, quad_opt.optimization_dt)

        '''
        waypoint_filename = 'trajectory_generation/waypoints/random_waypoints.csv'
        generate_random_waypoints(waypoint_filename, hsize=hsize, num_waypoints=num_waypoints)
        create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, quad_opt.optimization_dt)

        # trajectory has a specific time step that I do not respect here
        x_trajectory, t_trajectory = utils.load_trajectory(output_trajectory_filename)
        '''
       


    if args.trajectory == 2:
        # Circle trajectory
        radius = 50
        t_max = 30

        trajectory_generator.sample_circle_trajectory_accelerating(radius, v_max, t_max, quad_opt.optimization_dt)
        
        '''
        #trajectory_generator.sample_trajectory('random', v_max, a_max, quad_opt.optimization_dt)

        circle_trajectory_filename = 'trajectory_generation/trajectories/circle_trajectory.csv'
        generate_circle_trajectory_accelerating(circle_trajectory_filename, radius, v_max, t_max=t_max, dt=quad_opt.optimization_dt)
        # trajectory has a specific time step that I do not respect here
        x_trajectory, t_trajectory = utils.load_trajectory(circle_trajectory_filename)
        '''


    x_trajectory, t_trajectory = trajectory_generator.load_trajectory()


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



    # Set quad to start position
    quad.set_state(x)

    print(f'Duration of simulation={t_simulation}, Number of simulation steps={Nopt}')
    simulation_time = 0
    for i in tqdm(range(Nopt)):
        
        # Set the part of trajectory relevant for current time as the MPC reference
        x_ref = utils.get_reference_chunk(x_trajectory, i, quad_opt.n_nodes)
        yref, yref_N = quad_opt.set_reference_trajectory(x_ref)


        # I dont think I need to run optimization more times as with the case of new opt
        # TODO: Figure out why OCP gives different solutions both times it is run. warm start?
        x_opt_acados, w_opt_acados, t_cpu, cost_solution = quad_opt.run_optimization(x)
        u = w_opt_acados[0,:] # control to be applied to quad

        x_pred = quad_opt.discrete_dynamics(x, u, simulation_dt)
        x_pred = x_opt_acados[1,:]
        # Save nlp solution diagnostics
        solution_times.append(t_cpu)
        cost_solutions.append(cost_solution)

        # Odometry every MPC timestep (100ms)    
        x_odom.append(x)
        x_ref_odom.append(yref[0,:13])
        w_odom.append(u)
        t_odom.append(simulation_time)
        x_pred_odom.append(x_pred)

        control_time = 0
        # Simulate the quad plant with the optimal control until the next MPC optimization step is reached
        while control_time < quad_opt.optimization_dt: 
            # ----------- Simulate ----------------
            # Uses the optimization model to predict one step ahead, used for gp fitting
            x_pred = quad_opt.discrete_dynamics(x, u, simulation_dt, body_frame=True)
            # Control the quad with the most recent u for the whole control period (multiple simulation steps for one optimization)
            quad.update(u, simulation_dt)


            # ----------- Save simulation results ----------------
            x = np.array(quad.get_state(quaternion=True, stacked=True)) # state at the next optim step

            # Save model aerodrag for GP validation, useful only when payload=False
            x_body_for_drag = quad.get_state(quaternion=True, stacked=False, body_frame=False) # in world frame because get_aero_drag takes world frame velocity
            a_drag_body = quad.get_aero_drag(x_body_for_drag, body_frame=True)
            
            # Save simulation results
            x_world = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=False)) # World frame referential
            x_body = np.array(quad.get_state(quaternion=True, stacked=True, body_frame=True)) # Body frame referential

            # Save simulation results
            # Add current simulation results to list for dataset creation and visualisation
            x_sim.append(x_world)
            u_sim.append(u)
            x_sim_body.append(x_body)
            
            x_pred_sim.append(x_pred)
            yref_now = yref[0,:]
            yref_sim.append(yref_now)
            aero_drag_sim.append(a_drag_body)



            # Counts until the next MPC optimization step is reached
            control_time += simulation_dt
        # Counts until simulation is finished
        simulation_time += quad_opt.optimization_dt
    

    t = np.linspace(0, t_simulation, len(x_sim))

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

    # measured state
    data['p'] = x_sim[:,0:3]
    data['q'] = x_sim[:,3:7]
    data['v'] = x_sim[:,7:10]
    data['w'] = x_sim[:,10:13]

    data['x'] = x_sim
    
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
    data['x_pred'] = x_pred_sim

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

    
    save_dict(data, args.output)
    print(f'Saved simulated data to {args.output}')

    
    
if __name__ == '__main__':
    main()