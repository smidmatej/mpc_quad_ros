import os

import sys
import inspect

try:
    # Methods of this script are called from elsewhere
    import trajectory_generation.uav_trajectory as uav_trajectory
except ImportError:
    # This file is executed as main 
    import uav_trajectory as uav_trajectory
import numpy as np



class TrajectoryGenerator:
    def __init__(self):


        path_to_file = inspect.getfile(self.__class__)
        self.path_to_directory = os.path.dirname(path_to_file)
        #print(f'path_to_directory is {self.path_to_directory}')

        
        # For user defined waypoints
        self.user_waypoints_filename =  os.path.join(self.path_to_directory, 'waypoints/user_defined_waypoints.csv')
        # For generating random waypoints for a trajectory
        self.waypoints_filename = os.path.join(self.path_to_directory, 'waypoints/waypoints.csv')

        # genTrajectory script first creates a trajectory as a polynomial, this class then samples it
        self.polynom_filename = os.path.join(self.path_to_directory, 'polynomial_trajectory/polynomial_representation.csv')


        # The final trajectory is sampled and saved here
        self.sampled_trajectory_filename = os.path.join(self.path_to_directory, 'trajectories/trajectory_sampled.csv')


        


    def sample_circle_trajectory_accelerating(self, radius, v_max, t_max=10, dt=0.01, start_point=np.array([0.0, 0.0, 0.0])):
        ts = np.arange(0, t_max, dt)
        p = np.empty((len(ts), 3))
        v = np.empty((len(ts), 3))
        a = np.empty((len(ts), 3))
        w = np.empty((len(ts)))
        for i, t in zip(range(len(ts)), ts):

            # I have no idea why the 2 is needed here
            w[i] = (i+1)/float(len(ts)) * v_max/radius

            p[i, :] = np.array([radius * np.cos(w[i] * t), radius * np.sin(w[i] * t), 0]) + np.array([-radius, 0.0, 0.0]) + start_point
            v[i, :] = np.array([-radius*w[i] * np.sin(w[i] * t), radius*w[i] * np.cos(w[i] * t), 0])*2 # and also here
            a[i, :] = np.array([-radius*w[i]*w[i] * np.cos(w[i] * t), -radius*w[i]*w[i] * np.sin(w[i] * t), 0])*2*2

        #print(f'w = {w}')
        data = np.concatenate((ts.reshape(-1,1), p, v, a), axis=1)

        np.savetxt(self.sampled_trajectory_filename, data, fmt="%.6f", delimiter=",", header='t,x,y,z,vx,vy,vz,ax,ay,az')
                


    def sample_circle_trajectory(self, radius, v_max, t_max=10, dt=0.01):
        with open(self.sampled_trajectory_filename, "w") as f:
            f.write("t,x,y,z\n")

            for t in np.arange(0, t_max, dt):
                f.write("{},{},{},{},{}\n".format(t, radius * np.cos(v_max * t), radius * np.sin(v_max * t), 0))

    def generate_random_waypoints(self, hsize=10, num_waypoints=10, hover_first=False, start_point=np.array([0.0, 0.0, 0.0]), end_point=np.array([0.0, 0.0, 0.0])):
        # generate random waypoints in a cube centered around center_of_cube
        
        #print(f'Generating {num_waypoints} random waypoints saving them to {waypoints_filename}')
        waypoints = list()
        center_of_cube = np.array([0,0,1.5*hsize]) # Moved the center of the cube up so that the generated trajectories are above the ground plane
        waypoints.append(start_point)
        if hover_first:
            waypoints.append(np.array([0.0, 0.0, hsize])) # first rise up from the ground plane
        for i in range(num_waypoints):
            newWaypoint = np.random.uniform(-hsize, hsize, 3) + center_of_cube 
            waypoints.append(newWaypoint)

        #waypoints.append(np.array([0.0, 0.0, hsize])) # return above the ground plane
        #if end_point is not None:
            # If end_point is not None, then add it to the list of waypoints
            # Otherwise end on the last random waypoint
            # waypoints.append(end_point) 
        #waypoints.append(np.array([0.0, 0.0, 0.0]))
        self.write_waypoints_to_file(waypoints)
        



    def write_waypoints_to_file(self, waypoints):
        """
        Write waypoints to file. This exists to keep the same format even if the waypoints are generated in a different place.
        """
        np.savetxt(self.waypoints_filename, waypoints, fmt="%.6f", delimiter=",")




    def sample_trajectory(self, type, v_max, a_max, dt=0.01):
        
        assert type in ['static', 'random', 'line'], f'Invalid type {type}'
        if type == 'static':
            waypoints_filename = self.user_waypoints_filename
        else:
            waypoints_filename = self.waypoints_filename


        #print(f"Executing: {execution_path +  '/genTrajectory -i '+ waypoints_filename + ' -o ' + polynom_filename + ' --v_max ' + str(v_max) + ' --a_max ' + str(a_max)}")
        
        # Runs the genTrajectory script to generate a trajectory as a polynomial
        os.system(self.path_to_directory + '/genTrajectory -i '\
                + waypoints_filename + ' -o ' + self.polynom_filename \
                + ' --v_max ' + str(v_max) + ' --a_max ' + str(a_max))

        
        traj = uav_trajectory.Trajectory()

        #print("Loading polynomial representation of trajectory from file: {}".format(polynom_filename))

        # Reads the polynomial trajectory
        traj.loadcsv(self.polynom_filename)

        
        #print(f'Saving sampled trajectory to file: {output_trajectory_filename} with dt={dt}')

        # Samples the trajectory and saves it to a file
        self.save_evals_csv(traj, self.sampled_trajectory_filename, dt=dt)
        

    def save_evals_csv(self, traj, filename, dt=0.01):

        #traj.stretchtime(0.1)
        ts = np.arange(0, traj.duration, dt)
        evals = np.empty((len(ts), 15))
        for t, i in zip(ts, range(0, len(ts))):
            e = traj.eval(t)
            evals[i, 0:3]  = e.pos
            evals[i, 3:6]  = e.vel
            evals[i, 6:9]  = e.acc
        data = np.concatenate((ts.reshape(-1,1), evals), axis=1)
        np.savetxt(filename, data, fmt="%.6f", delimiter=",", header='t,x,y,z,vx,vy,vz,ax,ay,az')


    def load_trajectory(self):
        """
        Loads a trajectory from a .csv file.

        :param filename: path to the .csv file
        :return: a numpy array with the trajectory
        """
        #breakpoint()
        data = np.genfromtxt(self.sampled_trajectory_filename, delimiter=',')
        traj_t = data[:, 0]
        traj_pos = data[:, 1:4]
        traj_vel = data[:, 4:7]

        # csv does not contain orientation data
        traj_q = np.repeat(np.array([1,0,0,0]).reshape(1,-1), len(traj_t), axis=0).reshape(len(traj_t), 4)
        traj_r = np.repeat(np.array([0,0,0]).reshape(1,-1), len(traj_t), axis=0).reshape(len(traj_t), 3)

        # Acceleration data is redundant
        traj_a = data[:, 7:10]

        traj_x = np.concatenate((traj_pos, traj_q, traj_vel, traj_r), axis=1)
        return traj_x, traj_t
            

if __name__ == '__main__':
    TrajectoryGenerator()