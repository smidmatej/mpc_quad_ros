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
from functools import partial

class Animator:
    def __init__(self, trajectory_filename, result_animation_filename, desired_number_of_frames):
        self.trajectory_filename = trajectory_filename
        self.result_animation_filename = result_animation_filename
        self.desired_number_of_frames = desired_number_of_frames

        self.prepare_data()
        self.prepare_figure()
        self.create_animation()


    def create_animation(self):

        interval = 20 # 50 fps    
        self.number_of_frames = self.position.shape[0]
        print('Creating animation...')
        print(f'Number of frames: {self.number_of_frames}, fps: {1000/interval}, duration: {self.number_of_frames*interval/1000} s')

        self.pbar = tqdm(total=self.number_of_frames)
        use_color_map = True
        ani = animation.FuncAnimation(self.fig, partial(self.update, use_color_map=use_color_map), frames=self.number_of_frames, interval=interval, fargs=(use_color_map))           
        #pbar.close()


        ani.save(result_animation_filename)
        #ani.save('docs/drone_flight.gif')


   

    def update(self, i, use_color_map):
        # Current position
        
        #print(f'Frame: {i}')
        self.particle.set_data_3d(self.position[i,0], self.position[i,1], self.position[i,2])

        if use_color_map:
            # Plots the visited trajectory based on color. Computationaly expensive
            self.traj, = self.ax.plot(self.position[i:i+2,0], self.position[i:i+2,1], self.position[i:i+2,2], \
                                        color=plt.cm.jet(self.speed[i]/max(self.speed)), linewidth=0.8)
            
        # orientation
        self.vector_up.set_data_3d(np.array([self.position[i,0], self.position[i,0] + self.body_up[i,0]]), \
                np.array([self.position[i,1], self.position[i,1] + self.body_up[i,1]]), \
                np.array([self.position[i,2], self.position[i,2] + self.body_up[i,2]]))

        # Norm of veloctity to plot to a different ax
        self.v_traj.set_data(self.t[:i+1], self.speed[:i+1])

        if i < self.number_of_frames-1:
            # dp_norm is a diff, missing the last value
            self.dp_traj.set_data(self.t[:i+1], self.dp_norm[:i+1])


        for j in range(self.control.shape[1]):
            self.control_traj[j].set_data(self.t[:i+1], self.control[:i+1,j])
        
        self.pbar.update()



    def prepare_data(self):
        # Load data
        self.data_dict = load_dict(self.trajectory_filename)
        # Calculates how many datapoints to skip to get the desired number of frames
        skip = int(self.data_dict['x_odom'].shape[0]/self.desired_number_of_frames)

        # Load data 
        self.position = self.data_dict['x_odom'][::skip,:3]
        self.orientation = self.data_dict['x_odom'][::skip, 3:7]
        self.velocity = self.data_dict['x_odom'][::skip,7:10]
        self.control = self.data_dict['w_odom'][::skip,:]
        self.t = self.data_dict['t_odom'][::skip]


        self.speed = np.linalg.norm(self.velocity, axis=1)

        self.dp = np.diff(self.position, axis=0)/np.diff(self.t)[:,None]
        self.dp_norm = np.linalg.norm(self.dp, axis=1)


        # Limits of the plot for xlim, ylim, zlim
        # xlim=ylim=zlim
        self.min_lim = min(min(self.position[:,0]), min(self.position[:,1]), min(self.position[:,2]))
        self.max_lim = max(max(self.position[:,0]), max(self.position[:,1]), max(self.position[:,2]))

        # Up arrow us just for visualization, dont want it to be too big/small
        self.up_arrow_length = (self.max_lim - self.min_lim)/5
        self.body_up = np.array([v_dot_q(np.array([0, 0, self.up_arrow_length]), self.orientation[i,:]) for i in range(self.orientation.shape[0])])


    def prepare_figure(self):

        animation.writer = animation.writers['ffmpeg']
        plt.ioff() # Turn off interactive mode to hide rendering animations



        # Color scheme convert from [0,255] to [0,1]
        cs = [[x/256 for x in (8, 65, 92)], \
                [x/256 for x in (204, 41, 54)], \
                [x/256 for x in (118, 148, 159)], \
                [x/256 for x in (232, 197, 71)]] 


        plt.style.use('fast')
        sns.set_style("whitegrid")

        gs = gridspec.GridSpec(2, 2)
        self.fig = plt.figure(figsize=(10,10), dpi=100)

        self.ax = self.fig.add_subplot(gs[0:2,0], projection='3d')
        # Get rid of colored axes planes
        # First remove fill
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        self.ax.xaxis.pane.set_edgecolor('k')
        self.ax.yaxis.pane.set_edgecolor('k')
        self.ax.zaxis.pane.set_edgecolor('k')

        self.particle, = plt.plot([],[], marker='o', color=cs[1])
        self.vector_up, = plt.plot([],[], color=cs[3])
        self.traj, = plt.plot([],[], color=cs[0], alpha=0.5)



        self.ax.set_xlim((self.min_lim, self.max_lim))
        self.ax.set_ylim((self.min_lim, self.max_lim))
        self.ax.set_zlim((self.min_lim, self.max_lim))

        self.ax.set_xlabel('Position x [m]')
        self.ax.set_ylabel('Position y [m]')
        self.ax.set_zlabel('Position z [m]')
        self.ax.set_title('Quadcopter flight')


        self.ax_control = self.fig.add_subplot(gs[0,1])
        self.control_traj = [None]*4
        for i in range(self.control.shape[1]):
            self.control_traj[i], = self.ax_control.plot([], [], label='u'+str(i), c=cs[i])

        self.ax_control.set_xlim((min(self.t), max(self.t)))
        self.ax_control.set_ylim((0, 1))
        self.ax_control.set_xlabel('Time [s]')
        self.ax_control.set_ylabel('Control u ')
        self.ax_control.set_title('Control')
        self.ax_control.legend(('u0', 'u1', 'u2', 'u3'), loc='upper right')

        self.ax_speed = self.fig.add_subplot(gs[1,1])
        self.v_traj, = plt.plot([],[], color=cs[0])
        self.dp_traj, = plt.plot([],[], color=cs[0])
        self.ax_speed.set_xlim((min(self.t), max(self.t)))
        self.ax_speed.set_ylim((0, max(self.speed)))
        self.ax_speed.set_xlabel('Time [s]')
        self.ax_speed.set_ylabel('Speed [m/s]')
        self.ax_speed.set_title('Velocity magnitude')

        self.fig.tight_layout()









    


plt.show()

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    trajectory_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', 'simulated_trajectory.pkl')
    result_animation_filename = os.path.join(dir_path, '..', 'outputs', 'python_simulation', 'animations', 'my_animation.mp4')
    Animator(trajectory_filename, result_animation_filename, desired_number_of_frames=100)