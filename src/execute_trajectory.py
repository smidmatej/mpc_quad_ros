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
from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse
from utils import utils
from quad_opt import quad_optimizer
from utils.save_dataset import *
from trajectory_generation.generate_trajectory import generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

from trajectory_generation.TrajectoryGenerator import TrajectoryGenerator
import pickle
    
import argparse
    
from mpc_quad.src.gp.GP import *
from mpc_quad.src.gp.GPE import GPEnsemble

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

    if args.gpe:
        #ensemble_path = "gp/models/ensemble"
        ensemble_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'python_simulation', 'gp_models')
        gpe = GPEnsemble(3)
        gpe.load(ensemble_path)
    else:
        gpe = None



    explorer = Explorer(gpe)

    trajectory_generator = TrajectoryGenerator()

    v_max_limit = 30
    a_max_limit = 30

    # This musnt be faster than the quad is capable of
    # Max velocity and acceleration along the trajectory
    v_max = args.v_max
    a_max = args.a_max

    v_max = explorer.velocity_to_explore

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
    


    if args.trajectory == 0:
        # static trajectory
        trajectory_generator.sample_trajectory('static', v_max, a_max, quad_opt.optimization_dt)
        
    if args.trajectory == 1:
        # Generate trajectory as reference for the quadrotor
        # new trajectory
        hsize = 10
        num_waypoints = 10
        trajectory_generator.generate_random_waypoints(hsize, num_waypoints)
        trajectory_generator.sample_trajectory('random', v_max, a_max, quad_opt.optimization_dt)

    if args.trajectory == 2:
        # Circle trajectory
        radius = 50
        t_max = 30

        trajectory_generator.sample_circle_trajectory_accelerating(radius, v_max, t_max, quad_opt.optimization_dt)
        

    x_trajectory, t_trajectory = trajectory_generator.load_trajectory()
    simulation_length = max(t_trajectory) # Simulation duration for this script
    # Simulation runs for simulation_length seconds and MPC is calculated every quad_opt.optimization_dt
    Nopt = round(simulation_length/quad_opt.optimization_dt) # number of times MPC control is calculated steps

    # initial condition
    x0 = np.array([0,0,0] + [1,0,0,0] + [0,0,0] + [0,0,0])
    save_filepath = args.output

    simulate_trajectory(quad, quad_opt, x0, x_trajectory, simulation_length, Nopt, simulation_dt, save_filepath)
    


def simulate_trajectory(quad, quad_opt, x0, x_trajectory, simulation_length, Nopt, simulation_dt, save_filepath):

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



    # Set quad to start position
    #quad.set_state(x)
    # TODO: DO I NEED TO SET THE STATE OF THE QUAD HERE?

    print(f'Duration of simulation={simulation_length}, Number of simulation steps={Nopt}')
    simulation_time = 0
    for i in tqdm(range(Nopt)):
        
        if i >= int(Nopt/2):
            for ii in range(quad_opt.n_nodes):
                quad_opt.acados_ocp_solver.set(ii, 'p', np.array([0.0, 1.0]))

            if i == int(Nopt/2):
                print("Halfway there!")
                #quad_opt.acados_ocp.parameter_values = np.array([0.0, 0.0]) # initial position
                #json_file = '_acados_ocp.json'
                #quad_opt.acados_ocp_solver = AcadosOcpSolver(quad_opt.acados_ocp, json_file=json_file)
                
                
                print(quad_opt.acados_ocp.model.f_impl_expr)

        # Set the part of trajectory relevant for current time as the MPC reference
        x_ref = utils.get_reference_chunk(x_trajectory, i, quad_opt.n_nodes)
        yref, yref_N = quad_opt.set_reference_trajectory(x_ref)


        # I dont think I need to run optimization more times as with the case of new opt
        # TODO: Figure out why OCP gives different solutions both times it is run. warm start?
        x_opt_acados, w_opt_acados, t_cpu, cost_solution = quad_opt.run_optimization(x)
        u = w_opt_acados[0,:] # control to be applied to quad

        #x_pred = quad_opt.discrete_dynamics(x, u, simulation_dt)
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

    
    save_dict(data, save_filepath)
    print(f'Saved simulated data to {save_filepath}')


    
if __name__ == '__main__':
    main()