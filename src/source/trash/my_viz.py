
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from mpl_toolkits import mplot3d

import numpy as np

def plot_position3D(x_trajectory):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(x_trajectory[0,:], x_trajectory[1,:], x_trajectory[2,:], 'gray')
	plt.show()

def plot_state_over_time(x_trajectory):
	fig, axs = plt.subplots(2, 2)

	for i in range(0,3):
		axs[0,0].plot(x_trajectory[i,:])
	axs[0,0].set_title('Position')


	for i in range(3,7):
		axs[1,0].plot(x_trajectory[i,:])
	axs[1,0].set_title('Quaternion')

	for i in range(7,10):
		axs[0,1].plot(x_trajectory[i,:])
	axs[0,1].set_title('Velocity')

	for i in range(10,13):
		axs[1,1].plot(x_trajectory[i,:])
	axs[1,1].set_title('Angle rate')

	plt.show()

def plot_animated(x_trajectory):
	fig = plt.figure()
	
	world_rad = 5
	anim = lambda i : animate(i, x_trajectory, fig, world_rad)
	line_ani = FuncAnimation(fig, anim, interval=100,   
	                                   frames=x_trajectory.shape[0])
	plt.show()
	

def animate(i, x_trajectory, fig, world_rad):
	

	ax = fig.add_subplot(1, 2, 1, projection='3d')

	ax.clear()

	# +1 because of the trajectory 
	ax.plot3D(x_trajectory[0,:i+1], x_trajectory[1,:i+1], x_trajectory[2,:i+1], 'gray') 
	# current location
	ax.scatter(x_trajectory[0,i], x_trajectory[1,i], x_trajectory[2,i])

	ax.set_xlim([-world_rad, world_rad])
	ax.set_ylim([-world_rad, world_rad])
	ax.set_zlim([-world_rad, world_rad])

	'''
	
	ax = fig.add_subplot(1, 2, 2)

	#ax.clear()

	# +1 because of the trajectory 
	for stateidx in range(0,3):
		ax.scatter(i,x_trajectory[stateidx,i])

	#ax.set_xlim([0, len(x_trajectory[0,:])])
	#breakpoint()
	ax.set_ylim([np.amin(x_trajectory[0:3,:]), np.amin(x_trajectory[0:3,:])])
	'''

