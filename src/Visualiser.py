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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

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
from gp.GPE import GPEnsemble

class Visualiser:
    def __init__(self, trajectory_filename):
        self.trajectory_filename = trajectory_filename
        print(f'Visualising data in {self.trajectory_filename}')
        self.data_dict = load_dict(self.trajectory_filename)
        


    def create_animation(self, result_animation_filename, desired_number_of_frames, use_color_map=True, gif=False):
        self.desired_number_of_frames = desired_number_of_frames
        self.use_color_map = use_color_map
        self.prepare_animation_data()
        self.prepare_animation_figure()

        interval = 20 # 50 fps    
        self.number_of_frames = self.position.shape[0]
        print('Creating animation...')
        print(f'Number of frames: {self.number_of_frames}, fps: {1000/interval}, duration: {self.number_of_frames*interval/1000} s')

        self.pbar = tqdm(total=self.number_of_frames)

        def animate(i):
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


        ani = animation.FuncAnimation(self.fig, animate, frames=self.number_of_frames, interval=interval)          
        ani.save(result_animation_filename + '.mp4', writer='ffmpeg', fps=10, dpi=100)
        if gif:
            ani.save(result_animation_filename + '.gif', writer='imagemagick', fps=10, dpi=100)



    def create_rgp_animation(self, result_animation_filename, desired_number_of_frames, use_color_map=True, gif=False):
        self.desired_number_of_frames = desired_number_of_frames
        self.use_color_map = use_color_map
        self.prepare_rgp_animation_data()
        self.prepare_rgp_animation_figure()

        interval = 20 # 50 fps    
        self.number_of_frames = len(self.X)
        print('Creating RGP animation...')
        print(f'Number of frames: {self.number_of_frames}, fps: {1000/interval}, duration: {self.number_of_frames*interval/1000} s')

        self.pbar = tqdm(total=self.number_of_frames)

        def animate(i):   
            for d in range(3):
                self.scat_basis_vectors[d].set_offsets(np.array([self.X[i][d].ravel(), self.y[i][d].ravel()]).T)

            self.pbar.update()


        ani = animation.FuncAnimation(self.fig, animate, frames=self.number_of_frames, interval=interval)           
        ani.save(result_animation_filename + '.mp4', writer='ffmpeg', fps=10, dpi=100)
        if gif:
            ani.save(result_animation_filename + '.gif', writer='imagemagick', fps=10, dpi=100)


    
   


    def prepare_animation_data(self):

        # Calculates how many datapoints to skip to get the desired number of frames
        skip = int(self.data_dict['x_odom'].shape[0]/self.desired_number_of_frames)

        # Load data 
        self.position = self.data_dict['x_odom'][::skip,:3]
        self.orientation = self.data_dict['x_odom'][::skip, 3:7]
        self.velocity = self.data_dict['x_odom'][::skip,7:10]
        self.control = w[::skip,:]
        self.t = t[::skip]


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
        self.cs = [[x/256 for x in (8, 65, 92)], \
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

        self.particle, = plt.plot([],[], marker='o', color=self.cs[1])
        self.vector_up, = plt.plot([],[], color=self.cs[3])
        self.traj, = plt.plot([],[], color=self.cs[0], alpha=0.5)



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
            self.control_traj[i], = self.ax_control.plot([], [], label='u'+str(i), c=self.cs[i])

        self.ax_control.set_xlim((min(self.t), max(self.t)))
        self.ax_control.set_ylim((0, 1))
        self.ax_control.set_xlabel('Time [s]')
        self.ax_control.set_ylabel('Control u ')
        self.ax_control.set_title('Control')
        self.ax_control.legend(('u0', 'u1', 'u2', 'u3'), loc='upper right')

        self.ax_speed = self.fig.add_subplot(gs[1,1])
        self.v_traj, = plt.plot([],[], color=self.cs[0])
        self.dp_traj, = plt.plot([],[], color=self.cs[0])
        self.ax_speed.set_xlim((min(self.t), max(self.t)))
        self.ax_speed.set_ylim((0, max(self.speed)))
        self.ax_speed.set_xlabel('Time [s]')
        self.ax_speed.set_ylabel('Speed [m/s]')
        self.ax_speed.set_title('Velocity magnitude')

        self.fig.tight_layout()


    def prepare_rgp_animation_figure(self):

        animation.writer = animation.writers['ffmpeg']
        plt.ioff() # Turn off interactive mode to hide rendering animations

        # Color scheme convert from [0,255] to [0,1]
        self.cs = [[x/256 for x in (8, 65, 92)], \
                [x/256 for x in (204, 41, 54)], \
                [x/256 for x in (118, 148, 159)], \
                [x/256 for x in (232, 197, 71)]] 


        plt.style.use('fast')
        sns.set_style("whitegrid")

        gs = gridspec.GridSpec(1, 3)
        self.fig = plt.figure(figsize=(10,6), dpi=100)

        labels = ['x', 'y', 'z']
        self.ax = [None]*3
        self.scat_basis_vectors = [None]*3

        for d in range(3):
            self.ax[d] = self.fig.add_subplot(gs[d])
            self.scat_basis_vectors[d] = self.ax[d].scatter([], [], marker='o', label='Basis Vectors')
            self.ax[d].set_xlim(self.x_lim)
            self.ax[d].set_ylim(self.y_lim)

            self.ax[d].set_xlabel('Velocity [ms-1]')
            self.ax[d].set_ylabel('Drag acceleration [ms-2]')
            self.ax[d].set_title(f'RGP basis vectors in {labels[d]}')

        self.fig.tight_layout()


    def prepare_rgp_animation_data(self):

        # Calculates how many datapoints to skip to get the desired number of frames
        skip = int(self.data_dict['x_odom'].shape[0]/self.desired_number_of_frames)

        # Load data 
        self.X_array = self.data_dict['rgp_basis_vectors'][::skip]
        self.y_array = self.data_dict['rgp_params'][::skip]

        n_basis = self.X_array.shape[1]//3
        self.X = [None]*self.X_array.shape[0]
        self.y = [None]*self.X_array.shape[0]

        for i in range(self.X_array.shape[0]):

            self.X[i] = [self.X_array[i][d*n_basis:(d+1)*n_basis] for d in range(3)]
            self.y[i] = [self.y_array[i][d*n_basis:(d+1)*n_basis] for d in range(3)]

        x_min = 100000
        x_max = -100000
        y_min = 100000
        y_max = -100000
        for i in range(len(self.X)):
            x_min = min(x_min, min(min(self.X[i][0]), min(self.X[i][1]), min(self.X[i][2])))
            x_max = max(x_max, max(max(self.X[i][0]), max(self.X[i][1]), max(self.X[i][2])))
            y_max = max(y_max, max(max(self.y[i][0]), max(self.y[i][1]), max(self.y[i][2])))
            y_min = min(y_min, min(min(self.y[i][0]), min(self.y[i][1]), min(self.y[i][2])))
        
        self.x_lim = (x_min, x_max)
        self.y_lim = (y_min, y_max)


    def create_rgp_full_animation(self, result_animation_filename, desired_number_of_frames, use_color_map=True, gif=False):


        self.desired_number_of_frames = desired_number_of_frames
        self.use_color_map = use_color_map
        self.prepare_rgp_full_animation_data()
        self.prepare_rgp_full_animation_figure()

        interval = 20 # 50 fps    
        self.number_of_frames = len(self.X_basis)
        print('Creating RGP animation...')
        print(f'Number of frames: {self.number_of_frames}, fps: {1000/interval}, duration: {self.number_of_frames*interval/1000} s')

        self.pbar = tqdm(total=self.number_of_frames)

        def animate(i):

            for d in range(3):
                #
                self.scat_basis_vectors[d].set_offsets(np.array([self.X_basis[i][d], self.mu_g_t[i][d]]).T)

                # Inneficient way to take a [:i][d] slice, should have used a array.
                X_sample_array = np.array([float(self.X_sample[j][d]) for j in range(i+1)]).reshape(-1,1)
                y_sample_array = np.array([float(self.y_sample[j][d]) for j in range(i+1)]).reshape(-1,1)
                #breakpoint()
                print(f'sample: ({self.X_sample[i][d]}, {self.y_sample[i][d]})')

                self.scat_samples[d].set_offsets(np.concatenate((X_sample_array, y_sample_array), axis=1))
                self.rgp_mean_plot[d].set_data(self.X_query[d], self.y_query[i][d])
                self.fill_between_plots[d].remove()
                self.fill_between_plots[d] = self.ax[d].fill_between(self.X_query[d].reshape(-1),
                    self.y_query[i][d].reshape(-1) - 2*self.std_query[i][d], 
                    self.y_query[i][d].reshape(-1) + 2*self.std_query[i][d], color=self.cs[1], alpha=0.2)
            self.pbar.update()


        ani = animation.FuncAnimation(self.fig, animate, frames=self.number_of_frames, interval=interval)           
        ani.save(result_animation_filename + '.mp4', writer='ffmpeg', fps=10, dpi=100)
        if gif:
            ani.save(result_animation_filename + '.gif', writer='imagemagick', fps=10, dpi=100)



    def prepare_rgp_full_animation_figure(self):

        animation.writer = animation.writers['ffmpeg']
        plt.ioff() # Turn off interactive mode to hide rendering animations

        # Color scheme convert from [0,255] to [0,1]
        self.cs = [[x/256 for x in (8, 65, 92)], \
                [x/256 for x in (204, 41, 54)], \
                [x/256 for x in (118, 148, 159)], \
                [x/256 for x in (232, 197, 71)]] 


        plt.style.use('fast')
        sns.set_style("whitegrid")

        gs = gridspec.GridSpec(1, 3)
        self.fig = plt.figure(figsize=(10,6), dpi=100)

        labels = ['x', 'y', 'z']

        self.ax = [None]*3
        self.scat_basis_vectors = [None]*3
        self.scat_samples = [None]*3
        self.rgp_mean_plot = [None]*3
        self.fill_between_plots = [None]*3


        for d in range(3):
            self.ax[d] = self.fig.add_subplot(gs[d])

            self.scat_samples[d] = self.ax[d].scatter([], [], marker='.', color=self.cs[1], label='Samples')
            self.scat_basis_vectors[d] = self.ax[d].scatter([], [], marker='o', color=self.cs[2], label='Basis Vectors')
            self.rgp_mean_plot[d], = self.ax[d].plot([], [], '--', color=self.cs[0], label='E[g(x)]')
            
            self.fill_between_plots[d] = self.ax[d].fill_between([],
                [], 
                [], color=self.cs[3], alpha=0.2, label="2std")

            self.ax[d].set_xlim(self.x_lim[d])
            self.ax[d].set_ylim(self.y_lim[d])

            self.ax[d].set_xlabel('Velocity [ms-1]')
            self.ax[d].set_ylabel('Drag acceleration [ms-2]')
            self.ax[d].legend()
            self.ax[d].set_title(f'RGP basis vectors in {labels[d]}')

        self.fig.tight_layout()


    def prepare_rgp_full_animation_data(self):
        
        print("Preparing data...")
        # Calculates how many datapoints to skip to get the desired number of frames
        skip = len(self.data_dict['x_odom'])//self.desired_number_of_frames
        
        '''
        breakpoint()
        # Load data 
        X_array = self.data_dict['rgp_basis_vectors'][::skip,:,:]
        mu_array = self.data_dict['rgp_mu_g_t'][::skip,:,:]
        C_array = self.data_dict['rgp_C_g_t'][::skip,:,:,:]
        theta_array = self.data_dict['rgp_theta'][::skip,:,:]
        breakpoint()
        n_dims = 3
        n_samples = X_array.shape[0]

        self.X = [[None]*n_dims]*n_samples
        self.mu_g_t = [[None]*n_dims]*n_samples
        self.C_g_t = [[None]*n_dims]*n_samples
        self.theta = [[None]*n_dims]*n_samples
        self.y_query = [None]*n_samples
        self.std_query = [None]*n_samples

        for d in range(n_dims):
            for i in range(n_samples):
                self.X[i][d] = X_array[i, d, :]
                self.mu_g_t[i][d] = mu_array[i, d, :]
                self.C_g_t[i][d] = C_array[i, d, :, :]
                self.theta[i][d] = theta_array[i, d, :]
        '''
        self.X_basis = self.data_dict['rgp_basis_vectors'][::skip]
        self.mu_g_t = self.data_dict['rgp_mu_g_t'][::skip]
        self.C_g_t = self.data_dict['rgp_C_g_t'][::skip]
        self.theta = self.data_dict['rgp_theta'][::skip]

        self.X_sample = self.data_dict['v_body'][::skip]
        self.y_sample = self.data_dict['a_drag'][::skip]



        n_dims = 3
        n_samples = len(self.X_basis)

        self.y_query = [None]*n_samples
        self.std_query = [None]*n_samples

        
        x_min = [100000]*n_dims
        x_max = [-100000]*n_dims
        for d in range(n_dims):
            for i in range(n_samples):
                x_min[d] = min(x_min[d], min(self.X_basis[i][d]))
                x_max[d] = max(x_max[d], max(self.X_basis[i][d]))

        self.X_query = [np.linspace(x_min[d], x_max[d], 10) for d in range(n_dims)]

        print("Predicting...")
        pbar = tqdm(total=n_samples)
        for i in range(n_samples):
            self.rgpe = GPEnsemble.frombasisvectors(self.X_basis[i], self.mu_g_t[i], self.C_g_t[i], self.theta[i])
            self.y_query[i], self.std_query[i] = self.rgpe.predict(self.X_query, std=True)
            pbar.update(1)
        pbar.close()


        y_min = [100000]*n_dims
        y_max = [-100000]*n_dims
        for d in range(n_dims):
            for i in range(n_samples):

                y_min[d] = min(y_min[d], min(min(self.mu_g_t[i][d]), min(self.y_query[i][d]), min(self.y_sample[i][d]), -2*min(self.std_query[i][d])))
                y_max[d] = max(y_max[d], max(max(self.mu_g_t[i][d]), max(self.y_query[i][d]), max(self.y_sample[i][d]), 2*max(self.std_query[i][d])))
        

        y_lim_dif = [y_max[d]-y_min[d] for d in range(n_dims)]
        self.x_lim = [(x_min[d], x_max[d]) for d in range(n_dims)]
        self.y_lim = [(y_min[d] - np.sign(y_lim_dif[d])*y_lim_dif[d]/10, y_max[d] + np.sign(y_lim_dif[d])*y_lim_dif[d]/10) for d in range(n_dims)]


    @staticmethod
    def rms(x, axis=0):
        return np.sqrt(np.mean(x**2, axis=axis))

    def plot_data(self, filepath, show=True, save=True):
        
        x_world = np.stack(self.data_dict['x_odom'], axis=0)
        x_world_ref = np.stack(self.data_dict['x_ref'], axis=0)

        w = np.stack(self.data_dict['w_odom'], axis=0)
        t = np.stack(self.data_dict['t_odom'], axis=0)
        t_cpu = np.stack(self.data_dict['t_cpu'], axis=0)
        cost = np.stack(self.data_dict['cost_solution'], axis=0)



        v_norm = np.linalg.norm(x_world[:,7:10], axis=1)
        v_ref_norm = np.linalg.norm(x_world_ref[:,7:10], axis=1)

        e_pos_ref = x_world[:,0:3] - x_world_ref[:,0:3]
        #rms_pos_ref = np.sqrt(np.mean((e_pos_ref)**2, axis=1))
        rms_pos_ref = self.rms(e_pos_ref, 1)
        e_quat_ref = x_world[:,3:7] - x_world_ref[:,3:7]
        #rms_quat_ref = np.sqrt(np.mean((e_quat_ref)**2, axis=1))
        rms_quat_ref = self.rms(e_quat_ref, 1)
        e_vel_ref = x_world[:,7:10] - x_world_ref[:,7:10]
        #rms_vel_ref = np.sqrt(np.mean((e_vel_ref)**2, axis=1))
        rms_vel_ref = self.rms(e_vel_ref, 1)
        e_rate_ref = x_world[:,10:13] - x_world_ref[:,10:13]
        #rms_rate_ref = np.sqrt(np.mean((e_rate_ref)**2, axis=1))
        rms_rate_ref = self.rms(e_rate_ref, 1)

        rms_total_ref = np.sqrt(np.mean((x_world - x_world_ref)**2, axis=1))


        # Color scheme convert from [0,255] to [0,1]
        self.cs_u = [[x/256 for x in (8, 65, 92)], \
                [x/256 for x in (204, 41, 54)], \
                [x/256 for x in (118, 148, 159)], \
                [x/256 for x in (232, 197, 71)]] 

        self.cs_rgb = [[x/256 for x in (205, 70, 49)], \
                [x/256 for x in (105, 220, 158)], \
                [x/256 for x in (102, 16, 242)], \
                [x/256 for x in (7, 59, 58)]]

        plt.style.use('fast')
        sns.set_style("whitegrid")
        

        fig = plt.figure(figsize=(20, 10))
        #gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 1])
        gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])
        ax = [None]*12
        ax[0] = plt.subplot(gs[0, 0])
        ax[1] = plt.subplot(gs[0, 1])
        ax[2] = plt.subplot(gs[0, 2])
        ax[3] = plt.subplot(gs[0, 3])
        ax[4] = plt.subplot(gs[1, 0])
        ax[5] = plt.subplot(gs[1, 1])
        ax[6] = plt.subplot(gs[1, 2])
        ax[7] = plt.subplot(gs[1, 3])
        ax[8] = plt.subplot(gs[2, 0])
        ax[9] = plt.subplot(gs[2, 1])
        ax[10] = plt.subplot(gs[2, 2])
        ax[11] = plt.subplot(gs[2, 3])




        ax[0].plot(t, x_world[:,0], label='x', color=self.cs_rgb[0])
        ax[0].plot(t, x_world[:,1], label='y', color=self.cs_rgb[1])
        ax[0].plot(t, x_world[:,2], label='z', color=self.cs_rgb[2])
        ax[0].plot(t, x_world_ref[:,0], label='x_ref', color=self.cs_rgb[0], linestyle='dashed')
        ax[0].plot(t, x_world_ref[:,1], label='y_ref', color=self.cs_rgb[1], linestyle='dashed')
        ax[0].plot(t, x_world_ref[:,2], label='z_ref', color=self.cs_rgb[2], linestyle='dashed')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Position [m]')
        ax[0].set_title('Position')

        ax[1].plot(t, x_world[:,3], label='qw', color=self.cs_rgb[0])
        ax[1].plot(t, x_world[:,4], label='qx', color=self.cs_rgb[1])
        ax[1].plot(t, x_world[:,5], label='qy', color=self.cs_rgb[2])
        ax[1].plot(t, x_world[:,6], label='qz', color=self.cs_rgb[3])
        ax[1].plot(t, x_world_ref[:,3], label='qw_ref', color=self.cs_rgb[0], linestyle='dashed')
        ax[1].plot(t, x_world_ref[:,4], label='qx_ref', color=self.cs_rgb[1], linestyle='dashed')
        ax[1].plot(t, x_world_ref[:,5], label='qy_ref', color=self.cs_rgb[2], linestyle='dashed')
        ax[1].plot(t, x_world_ref[:,6], label='qz_ref', color=self.cs_rgb[3], linestyle='dashed')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Quaternion')
        ax[1].set_title('Orientation')

        ax[2].plot(t, x_world[:,7], label='vx', color=self.cs_rgb[0])
        ax[2].plot(t, x_world[:,8], label='vy', color=self.cs_rgb[1])
        ax[2].plot(t, x_world[:,9], label='vz', color=self.cs_rgb[2])
        ax[2].plot(t, x_world_ref[:,7], label='vx_ref', color=self.cs_rgb[0], linestyle='dashed')
        ax[2].plot(t, x_world_ref[:,8], label='vy_ref', color=self.cs_rgb[1], linestyle='dashed')
        ax[2].plot(t, x_world_ref[:,9], label='vz_ref', color=self.cs_rgb[2], linestyle='dashed')

        ax[2].plot(t, v_norm, label='v_norm', color=self.cs_rgb[3])
        ax[2].plot(t, v_ref_norm, label='v_ref_norm', color=self.cs_rgb[3], linestyle='dashed')
        ax[2].set_title('Velocity')
        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylabel('Velocity [m/s]')

        ax[3].plot(t, x_world[:,10], label='wx', color=self.cs_rgb[0])
        ax[3].plot(t, x_world[:,11], label='wy', color=self.cs_rgb[1])
        ax[3].plot(t, x_world[:,12], label='wz', color=self.cs_rgb[2])
        ax[3].plot(t, x_world_ref[:,10], label='wx_ref', color=self.cs_rgb[0], linestyle='dashed')
        ax[3].plot(t, x_world_ref[:,11], label='wy_ref', color=self.cs_rgb[1], linestyle='dashed')
        ax[3].plot(t, x_world_ref[:,12], label='wz_ref', color=self.cs_rgb[2], linestyle='dashed')
        ax[3].set_title('Angular Velocity')
        ax[3].set_xlabel('Time [s]')
        ax[3].set_ylabel('Angular Velocity [rad/s]')


        ax[4].plot(t, e_pos_ref[:,0], label='e_x', color=self.cs_rgb[0])
        ax[4].plot(t, e_pos_ref[:,1], label='e_y', color=self.cs_rgb[1])
        ax[4].plot(t, e_pos_ref[:,2], label='e_z', color=self.cs_rgb[2])
        ax[4].plot(t, rms_pos_ref, label="rms", color=self.cs_rgb[3])
        ax[4].set_title(f"RMS Position Error, Total: {self.rms(rms_pos_ref, 0)*1e3:.2f}mm")
        ax[4].set_xlabel('Time [s]')
        ax[4].set_ylabel('RMS Position Error [m]')
        ax[4].legend()


        ax[5].plot(t, rms_quat_ref, label='rms', color=self.cs_rgb[0])
        ax[5].set_title(f'RMS Quaternion Error')
        ax[5].set_xlabel('Time [s]')
        ax[5].set_ylabel('RMS Quaternion Error [m/s]')

        ax[6].plot(t, e_vel_ref[:,0], label='e_vx', color=self.cs_rgb[0])
        ax[6].plot(t, e_vel_ref[:,1], label='e_vy', color=self.cs_rgb[1])
        ax[6].plot(t, e_vel_ref[:,2], label='e_vz', color=self.cs_rgb[2])
        ax[6].plot(t, rms_vel_ref, label='rms', color=self.cs_rgb[3])
        ax[6].set_title(f'RMS Velocity Error, Total: {self.rms(rms_vel_ref, 0)*1000:.2f}mm/s')
        ax[6].set_xlabel('Time [s]')
        ax[6].set_ylabel('RMS Velocity Error [m/s]')
        ax[6].legend()

        ax[7].plot(t, e_rate_ref[:,0], label='e_vx', color=self.cs_rgb[0])
        ax[7].plot(t, e_rate_ref[:,1], label='e_vy', color=self.cs_rgb[1])
        ax[7].plot(t, e_rate_ref[:,2], label='e_vz', color=self.cs_rgb[2])
        ax[7].plot(t, rms_rate_ref, label='rms', color=self.cs_rgb[3])
        ax[7].set_title(f'RMS Angular Velocity Error')
        ax[7].set_xlabel('Time [s]')
        ax[7].set_ylabel('RMS Angular Velocity Error [rad/s]')
        ax[7].legend()


        ax[8].plot(x_world[:,7], e_pos_ref[:,0], label='e_vx', color=self.cs_rgb[0])
        ax[8].plot(x_world[:,8], e_pos_ref[:,1], label='e_vy', color=self.cs_rgb[1])
        ax[8].plot(x_world[:,9], e_pos_ref[:,2], label='e_vz', color=self.cs_rgb[2])
        ax[8].plot(v_norm, rms_pos_ref, label='rms', color=self.cs_rgb[3])
        ax[8].set_xlabel('Velocity [m/s]')
        ax[8].set_ylabel('Position Error [m]')
        ax[8].set_title('Position error as a function of velocity')
        ax[8].legend()

        ax[9].plot(t, w[:,0], label='u1', color=self.cs_u[0])
        ax[9].plot(t, w[:,1], label='u2', color=self.cs_u[1])
        ax[9].plot(t, w[:,2], label='u3', color=self.cs_u[2])
        ax[9].plot(t, w[:,3], label='u4', color=self.cs_u[3])
        ax[9].set_xlabel('Time [s]')
        ax[9].set_ylabel('Control Input')
        ax[9].set_title('Control Input')

        ax[10].plot(t, t_cpu[:]*1e3, label='t_cpu', color=self.cs_rgb[0])
        ax[10].set_xlabel('Time [s]')
        ax[10].set_ylabel('CPU Time [ms]')
        ax[10].set_title('MPC CPU Time')

        ax[11].plot(t, cost[:], label='solution_cost', color=self.cs_rgb[0])
        ax[11].set_xlabel('Time [s]')
        ax[11].set_ylabel('Solution Cost')
        ax[11].set_title('Solution Cost')

        
        plt.tight_layout()

        if save:
            plt.savefig(filepath, format="pdf", bbox_inches="tight")

        # Show needs to come after savefig because it clears the figure
        if show:
            plt.show()



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    simulation = 'gazebo_simulation'
    simulation = 'python_simulation'
    
    
    trajectory_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'data', 'simulated_trajectory.pkl')
    result_animation_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'animations', 'my_animation.mp4')
    result_plot_filename = os.path.join(dir_path, '..', 'outputs', simulation, 'img', 'trajectory.pdf')
    visualiser = Visualiser(trajectory_filename)
    visualiser.create_animation(result_animation_filename, 100, True)
    #visualiser.plot_data(result_plot_filename)
