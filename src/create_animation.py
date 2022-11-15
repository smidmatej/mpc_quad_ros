from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from gp.data_loader import data_loader
from utils.utils import v_dot_q
import matplotlib as style
from matplotlib import gridspec
import matplotlib.colors as colors
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import os
from utils.save_dataset import load_dict

def update(i):
    #particle.set_data(x[i],y[i])
    #print(pbar)
    # Current position
    particle.set_data_3d(position[i,0], position[i,1], position[i,2])

    # Visited positions
    traj, = ax.plot(position[i:i+2,0], position[i:i+2,1], position[i:i+2,2], color=plt.cm.jet(speed[i]/max(speed)), linewidth=0.8)

    # orientation
    vector_up.set_data_3d(np.array([position[i,0], position[i,0] + body_up[i,0]]), \
            np.array([position[i,1], position[i,1] + body_up[i,1]]), \
            np.array([position[i,2], position[i,2] + body_up[i,2]]))

    # Norm of veloctity to plot to a different ax
    v_traj.set_data(t[:i+1], speed[:i+1])

    if i < number_of_frames-1:
        # dp_norm is a diff, missing the last value
        dp_traj.set_data(t[:i+1], dp_norm[:i+1])


    for j in range(control.shape[1]):
        control_traj[j].set_data(t[:i+1], control[:i+1,j])
    
    global pbar
    pbar.update()

    return particle,traj,v_traj





plt.style.use('fast')
sns.set_style("whitegrid")

dir_path = os.path.dirname(os.path.realpath(__file__))
trajectory_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', 'simulated_trajectory.pkl')
result_animation_filename = os.path.join(dir_path, '..', 'outputs', 'python_simulation', 'animations', 'my_animation.mp4')

# an estimate 
desired_number_of_frames = 100

data_dict = load_dict(trajectory_filename)

# Calculates how many datapoints to skip to get the desired number of frames
skip = int(data_dict['x_odom'].shape[0]/desired_number_of_frames)

# Load data 
position = data_dict['x_odom'][::skip,:3]
orientation = data_dict['x_odom'][::skip, 3:7]
velocity = data_dict['x_odom'][::skip,7:10]
control = data_dict['w_odom'][::skip,:]
t = data_dict['t_odom'][::skip]


# Limits of the plot for xlim, ylim, zlim
# xlim=ylim=zlim
min_lim = min(min(position[:,0]), min(position[:,1]), min(position[:,2]))
max_lim = max(max(position[:,0]), max(position[:,1]), max(position[:,2]))


speed = np.linalg.norm(velocity, axis=1)

dp = np.diff(position, axis=0)/np.diff(t)[:,None]
dp_norm = np.linalg.norm(dp, axis=1)

# Up arrow us just for visualization, dont want it to be too big/small
up_arrow_length = (max_lim - min_lim)/5
body_up = np.array([v_dot_q(np.array([0,0,up_arrow_length]), orientation[i,:]) for i in range(orientation.shape[0])])




    
animation.writer = animation.writers['ffmpeg']
plt.ioff() # Turn off interactive mode to hide rendering animations


# Color scheme convert from [0,255] to [0,1]
cs = [[x/256 for x in (8, 65, 92)], \
        [x/256 for x in (204, 41, 54)], \
        [x/256 for x in (118, 148, 159)], \
        [x/256 for x in (232, 197, 71)]] 


gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10,10), dpi=100)

ax = fig.add_subplot(gs[0:2,0], projection='3d')
# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')

particle, = plt.plot([],[], marker='o', color=cs[1])
vector_up, = plt.plot([],[], color=cs[3])

traj, = plt.plot([],[], color=cs[0], alpha=0.5)



ax.set_xlim((min_lim, max_lim))
ax.set_ylim((min_lim, max_lim))
ax.set_zlim((min_lim, max_lim))

ax.set_xlabel('Position x [m]')
ax.set_ylabel('Position y [m]')
ax.set_zlabel('Position z [m]')
ax.set_title('Quadcopter flight')


ax_control = fig.add_subplot(gs[0,1])
control_traj = [None]*4
for i in range(control.shape[1]):
    control_traj[i], = ax_control.plot([], [], label='u'+str(i), c=cs[i])

ax_control.set_xlim((min(t), max(t)))
ax_control.set_ylim((0, 1))
ax_control.set_xlabel('Time [s]')
ax_control.set_ylabel('Control u ')
ax_control.set_title('Control')
ax_control.legend(('u0', 'u1', 'u2', 'u3'), loc='upper right')

ax_speed = fig.add_subplot(gs[1,1])
v_traj, = plt.plot([],[], color=cs[0])
dp_traj, = plt.plot([],[], color=cs[0])
ax_speed.set_xlim((min(t), max(t)))
ax_speed.set_ylim((0, max(speed)))
ax_speed.set_xlabel('Time [s]')
ax_speed.set_ylabel('Speed [m/s]')
ax_speed.set_title('Velocity magnitude')



fig.tight_layout()


interval = 20 # 50 fps    
number_of_frames = position.shape[0]
print('Creating animation...')
print(f'Number of frames: {number_of_frames}, fps: {1000/interval}, duration: {number_of_frames*interval/1000} s')

pbar = tqdm(total=number_of_frames)
    #pbar.update()
ani = animation.FuncAnimation(fig, update, frames=number_of_frames, interval=interval)           
#pbar.close()


ani.save(result_animation_filename)
#ani.save('docs/drone_flight.gif')

plt.show()
