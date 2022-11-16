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

class Visualiser:
    def __init__(self, trajectory_filename):
        self.trajectory_filename = trajectory_filename
        self.data_dict = load_dict(self.trajectory_filename)
        


    def create_animation(self, result_animation_filename, desired_number_of_frames, use_color_map=True):
        self.desired_number_of_frames = desired_number_of_frames
        self.use_color_map = use_color_map
        self.prepare_animation_data()
        self.prepare_animation_figure()

        interval = 20 # 50 fps    
        self.number_of_frames = self.position.shape[0]
        print('Creating animation...')
        print(f'Number of frames: {self.number_of_frames}, fps: {1000/interval}, duration: {self.number_of_frames*interval/1000} s')

        self.pbar = tqdm(total=self.number_of_frames)

        ani = animation.FuncAnimation(self.fig, self.update, frames=self.number_of_frames, interval=interval)           
        #pbar.close()


        ani.save(result_animation_filename)
        #ani.save('docs/drone_flight.gif')


   

    def update(self, i):
        # Current position
        
        #print(f'Frame: {i}')
        self.particle.set_data_3d(self.position[i,0], self.position[i,1], self.position[i,2])

        if self.use_color_map:
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



    def prepare_animation_data(self):

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


    def prepare_animation_figure(self):

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


    def plot_data(self, filepath):

        print(f'Data dict keys: {self.data_dict.keys()}')
        v_norm = np.linalg.norm(self.data_dict['x_odom'][:,7:10], axis=1)
        v_ref_norm = np.linalg.norm(self.data_dict['x_ref'][:,7:10], axis=1)

        rms_pos_ref = np.sqrt(np.mean((self.data_dict['x_odom'][:,0:3] - self.data_dict['x_ref'][:,0:3])**2, axis=1))
        rms_vel_ref = np.sqrt(np.mean((self.data_dict['x_odom'][:,7:10] - self.data_dict['x_ref'][:,7:10])**2, axis=1))
        rms_quat_ref = np.sqrt(np.mean((self.data_dict['x_odom'][:,3:7] - self.data_dict['x_ref'][:,3:7])**2, axis=1))
        rms_rate_ref = np.sqrt(np.mean((self.data_dict['x_odom'][:,10:13] - self.data_dict['x_ref'][:,10:13])**2, axis=1))

        rms_total_ref = np.sqrt(np.mean((self.data_dict['x_odom'] - self.data_dict['x_ref'])**2, axis=1))


        # Color scheme convert from [0,255] to [0,1]
        cs_u = [[x/256 for x in (8, 65, 92)], \
                [x/256 for x in (204, 41, 54)], \
                [x/256 for x in (118, 148, 159)], \
                [x/256 for x in (232, 197, 71)]] 

        cs_rgb = [[x/256 for x in (205, 70, 49)], \
                [x/256 for x in (105, 220, 158)], \
                [x/256 for x in (102, 16, 242)], \
                [x/256 for x in (7, 59, 58)]]

        plt.style.use('fast')
        sns.set_style("whitegrid")
        

        fig = plt.figure(figsize=(20, 10))
        #gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 1])
        gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[0, 3])
        ax5 = plt.subplot(gs[1, 0])
        ax6 = plt.subplot(gs[1, 1])
        ax7 = plt.subplot(gs[1, 2])
        ax8 = plt.subplot(gs[1, 3])
        ax9 = plt.subplot(gs[2, 0])
        ax10 = plt.subplot(gs[2, 1])
        ax11 = plt.subplot(gs[2, 2])
        ax12 = plt.subplot(gs[2, 3])
        



        ax1.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,0], label='x', color=cs_rgb[0])
        ax1.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,1], label='y', color=cs_rgb[1])
        ax1.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,2], label='z', color=cs_rgb[2])
        ax1.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,0], label='x_ref', color=cs_rgb[0], linestyle='dashed')
        ax1.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,1], label='y_ref', color=cs_rgb[1], linestyle='dashed')
        ax1.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,2], label='z_ref', color=cs_rgb[2], linestyle='dashed')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position [m]')
        ax1.set_title('Position')

        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,3], label='qw', color=cs_rgb[0])
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,4], label='qx', color=cs_rgb[1])
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,5], label='qy', color=cs_rgb[2])
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,6], label='qz', color=cs_rgb[3])
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,3], label='qw_ref', color=cs_rgb[0], linestyle='dashed')
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,4], label='qx_ref', color=cs_rgb[1], linestyle='dashed')
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,5], label='qy_ref', color=cs_rgb[2], linestyle='dashed')
        ax2.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,6], label='qz_ref', color=cs_rgb[3], linestyle='dashed')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Quaternion')
        ax2.set_title('Orientation')

        ax3.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,7], label='vx', color=cs_rgb[0])
        ax3.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,8], label='vy', color=cs_rgb[1])
        ax3.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,9], label='vz', color=cs_rgb[2])
        ax3.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,7], label='vx_ref', color=cs_rgb[0], linestyle='dashed')
        ax3.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,8], label='vy_ref', color=cs_rgb[1], linestyle='dashed')
        ax3.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,9], label='vz_ref', color=cs_rgb[2], linestyle='dashed')

        ax3.plot(self.data_dict['t_odom'], v_norm, label='v_norm', color=cs_rgb[3])
        ax3.plot(self.data_dict['t_odom'], v_ref_norm, label='v_ref_norm', color=cs_rgb[3], linestyle='dashed')
        ax3.set_title('Velocity')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Velocity [m/s]')

        ax4.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,10], label='wx', color=cs_rgb[0])
        ax4.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,11], label='wy', color=cs_rgb[1])
        ax4.plot(self.data_dict['t_odom'], self.data_dict['x_odom'][:,12], label='wz', color=cs_rgb[2])
        ax4.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,10], label='wx_ref', color=cs_rgb[0], linestyle='dashed')
        ax4.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,11], label='wy_ref', color=cs_rgb[1], linestyle='dashed')
        ax4.plot(self.data_dict['t_odom'], self.data_dict['x_ref'][:,12], label='wz_ref', color=cs_rgb[2], linestyle='dashed')
        ax4.set_title('Angular Velocity')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Angular Velocity [rad/s]')


        ax5.plot(self.data_dict['t_odom'], rms_pos_ref, label='rms_pos_ref', color=cs_rgb[0])
        ax5.set_title('RMS Position Error')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('RMS Position Error [m]')

        ax6.plot(self.data_dict['t_odom'], rms_quat_ref, label='rms_quat_ref', color=cs_rgb[0])
        ax6.set_title('RMS Quaternion Error')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('RMS Quaternion Error [m/s]')

        ax7.plot(self.data_dict['t_odom'], rms_vel_ref, label='rms_vel_ref', color=cs_rgb[0])
        ax7.set_title('RMS Velocity Error')
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel('RMS Velocity Error [m/s]')

        ax8.plot(self.data_dict['t_odom'], rms_rate_ref, label='rms_rate_ref', color=cs_rgb[0])
        ax8.set_title('RMS Angular Velocity Error')
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('RMS Angular Velocity Error [rad/s]')


        ax9.plot(self.data_dict['t_odom'], self.data_dict['w_odom'][:,0], label='u1', color=cs_u[0])
        ax9.plot(self.data_dict['t_odom'], self.data_dict['w_odom'][:,1], label='u2', color=cs_u[1])
        ax9.plot(self.data_dict['t_odom'], self.data_dict['w_odom'][:,2], label='u3', color=cs_u[2])
        ax9.plot(self.data_dict['t_odom'], self.data_dict['w_odom'][:,3], label='u4', color=cs_u[3])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel('Control Input')
        ax9.set_title('Control Input')


        print(type(self.data_dict['t_cpu'][:]))
        ax10.plot(self.data_dict['t_cpu'][:]*1e3, label='t_cpu', color=cs_rgb[0])
        ax10.set_xlabel('Time [s]')
        ax10.set_ylabel('CPU Time [ms]')
        ax10.set_title('MPC CPU Time')

        ax11.plot(self.data_dict['cost_solution'][:], label='solution_cost', color=cs_rgb[0])
        ax11.set_xlabel('Time [s]')
        ax11.set_ylabel('Solution Cost')
        ax11.set_title('Solution Cost')

        
        plt.tight_layout()

        plt.savefig(filepath, format="pdf", bbox_inches="tight")

        # Show needs to come after savefig because it clears the figure
        plt.show()



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    trajectory_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', 'simulated_trajectory.pkl')
    result_animation_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'animations', 'my_animation.mp4')
    result_plot_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'img', 'trajectory.pdf')
    visualiser = Visualiser(trajectory_filename)
    #animator.create_animation(result_animation_filename, 100, True)
    visualiser.plot_data(result_plot_filename)