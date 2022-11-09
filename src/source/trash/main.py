import quad
from tqdm import tqdm

import pdb

import numpy as np
#from visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results
import my_viz



from time import sleep
import os
#os.environ["PYTHONBREAKPOINT"] = "0"


real_time_artists = None
    
# Initialize real time plot stuff



if __name__ == "__main__":
	my_quad = quad.Quadrotor3D()
	
	

	simulation_steps = 20000
	simulation_step_length = 0.1
	x_history = list()
	np.random.seed(0)
	t = np.arange(0, simulation_steps*simulation_step_length, simulation_step_length)

	quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
	quad_trajectory = np.zeros((13, simulation_steps))
	#print(my_quad.get_state(quaternion=True))
	#print(range(t))
	for current_idx in tqdm(np.arange(1,simulation_steps)):
		u = np.random.random((4,))
		u = np.array([1,1,1,1])*10
		#print(u)
		my_quad.update(u, simulation_step_length)

		#breakpoint()
		#x_pred = np.zeros((13,1))
		quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
		quad_trajectory[:, current_idx] = quad_current_state

		
		#x_history.append(my_quad.get_state(quaternion=True))

	
	#my_viz.plot_position3D(quad_trajectory)

	#my_viz.plot_state_over_time(quad_trajectory)
	my_viz.plot_animated(quad_trajectory)


