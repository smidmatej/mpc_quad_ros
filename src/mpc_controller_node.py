#!/usr/bin/env python3 

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


import rospy

import time

from geometry_msgs.msg import Pose, Point
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
'''
# ControlCommand control_mode field
uint8 NONE=0
uint8 ATTITUDE=1
uint8 BODY_RATES=2
uint8 ANGULAR_ACCELERATIONS=3
uint8 ROTOR_THRUSTS=4
'''
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Bool, Empty

from visualization_msgs.msg import Marker

# for path visualization
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped 
from tqdm import tqdm

from gp.gp_train import train_gp
from Explorer import Explorer

from mpcros.msg import Trajectory
from mpcros.msg import Trajectory_request

import std_msgs
import numpy as np
import os
import sys
from typing import Tuple

from quad import Quadrotor3D 
from quad_opt import quad_optimizer
from gp.GPE import GPEnsemble

from Logger import Logger

from utils.utils import get_reference_chunk, v_dot_q, get_reference_chunk, quaternion_to_euler, rospy_time_to_float, quaternion_inverse

class MPC_controller:
        
    def __init__(self):

        self.quad_name = 'hummingbird'
        rospy.init_node('controller')

        self.v_max = rospy.get_param('/mpcros/mpc_controller/v_max')
        self.a_max = rospy.get_param('/mpcros/mpc_controller/a_max')
        self.trajectory_type = rospy.get_param('/mpcros/mpc_controller/trajectory_type') # node_name/argsname
        self.training_run = rospy.get_param('/mpcros/mpc_controller/training') # node_name/argsname
        self.training_trajectories_count = rospy.get_param('/mpcros/mpc_controller/training_trajectories_count') # node_name/argsname
        self.use_gp = rospy.get_param('/mpcros/mpc_controller/use_gp')
        self.gp_path = rospy.get_param('/mpcros/mpc_controller/gp_path')
        self.gp_from_file = rospy.get_param('/mpcros/mpc_controller/gp_from_file')
        self.n_basis_vectors = rospy.get_param('/mpcros/mpc_controller/n_basis_vectors')
        self.explore = rospy.get_param('/mpcros/mpc_controller/explore')
        self.t_lookahead = rospy.get_param('/mpcros/mpc_controller/t_lookahead')
        self.n_nodes = rospy.get_param('/mpcros/mpc_controller/n_nodes')

        self.dir_path = os.path.dirname(os.path.realpath(__file__))


        # --------------------- Logging ---------------------
        if self.training_run:
            self.trajectory_type = 'random'
            log_filename = f"training_v{self.v_max:.0f}_a{self.a_max:.0f}_gp{self.use_gp}"
            if self.explore:
                log_filename = "explore" 
        else:
            log_filename = f"test_{self.trajectory_type}_v{self.v_max:.0f}_a{self.a_max:.0f}_gp{self.use_gp}"
        
        self.logger = Logger(log_filename)

        
        # --------------------- Constants ---------------------
        # Odometry is published with a average frequency of 100 Hz
        # TODO: check if this is the case, sometimes it is delayed a bit
        self.ODOMETRY_DT = 1/100
        # TODO: Gazebo runs with a variable rate, this changes the odometry dt, but the trajectories are still sampled the same. This is clearly wrong.
        self.EPSILON_TRAJECTORY_FINISHED = 1 # Distance to consider a trajectory finished [m]
        self.HOVER_POS = np.array([0.0, 0.0, 3.0])


        # --------------------- Variables ---------------------
        self.trajectory_ready = False # Flag for the trajectory request callback
        self.need_trajectory_to_hover = True # Flag for request new trajectory to hover
        self.number_of_trajectories_finished = 0 # Counts the number of finished trajectories

        self.doing_a_line = False # This is a hack to not count the first trajectory into the number of finished trajectories
        
        self.rebooting_controller = False # Flag for rebooting the controller
        self.rebooted_controller = False # Flag for rebooting the controller
        self.last_reboot_timestamp = -1.0 # Timestamp of the last reboot
        self.pbar = None # Progress bar for trajectory following

   
        # --------------------- Topics ---------------------
        self.reference_trajectory_topic = "reference/trajectory"
        self.new_trajectory_request_topic = "reference/new_trajectory_request"
        self.pose_topic = "/" + self.quad_name + "/ground_truth/odometry"
        self.autopilot_pose_topic = "/" + self.quad_name  +'/autopilot/pose_command'
        self.control_topic = "/" + self.quad_name + "/autopilot/control_command_input"
        self.marker_topic = "rviz/marker"
        self.reference_path_chunk_topic = "rviz/reference_chunk"
        self.optimal_path_topic = "rviz/optimal_path"
        self._force_hover__topic = "/" + self.quad_name + "/autopilot/force_hover"

        '''
        if self.explore:
            # Let the explorer decide on the maximum velocity
            self.explorer = Explorer(self.quad_opt.gpe)
            self.v_max = self.explorer.velocity_to_explore
            self.a_max = self.explorer.velocity_to_explore
        '''

        rospy.logwarn("Initializing MPC controller")

        self.initialize_MPC()

        # --------------------- Publishers ---------------------
        self.optimal_path_pub = rospy.Publisher(self.optimal_path_topic, Path, queue_size=1) # Path from current quad position onto the path
        self.reference_path_chunk_pub = rospy.Publisher(self.reference_path_chunk_topic, Path, queue_size=1) # Chunk of the reference path that is used for MPC
        self.markerPub = rospy.Publisher(self.marker_topic, Marker, queue_size=10)      
        self.new_trajectory_request_pub = rospy.Publisher(self.new_trajectory_request_topic, Trajectory_request, queue_size=1)
        self._go_to_pose_pub = rospy.Publisher(self.autopilot_pose_topic, PoseStamped, queue_size=1)
        self._force_hover_pub = rospy.Publisher(self._force_hover__topic, Empty,queue_size=1)
        self.actuator_publisher = rospy.Publisher(self.control_topic, ControlCommand, queue_size=1, tcp_nodelay=True)
        # --------------------- Subscribers ---------------------
        self.trajectory_sub = rospy.Subscriber(self.reference_trajectory_topic, Trajectory, self.trajectory_received_cb) # Reference trajectory
        self.pose_subscriber = rospy.Subscriber(self.pose_topic, Odometry, self.pose_received_cb) # Pose is published by the simulator at 100 Hz!
            
        rospy.logwarn("MPC controller initilized")



    def initialize_MPC(self):
        '''
        Initializes the MPC controller from the current self parameters
        '''

        # ---- Quadrotor model ----
        # Instantiate quadrotor model with default parameters
        quad = Quadrotor3D(payload=False, drag=True) # Controlled plant s
        # Loads parameters of  a quad from a xarco file into quad object
        dir_path = os.path.dirname(os.path.realpath(__file__))
        params_filepath = os.path.join(dir_path, '..' , 'config', self.quad_name + '.xacro')
        quad.set_parameters_from_file(params_filepath, self.quad_name)

        # ---- GPE ----
        self.ensemble_path = os.path.join(self.dir_path, self.gp_path)

        if self.use_gp == 0:
            print("Not using GPE")
            gpe = None
        elif self.use_gp == 1:
            gpe = GPEnsemble.fromdir(self.ensemble_path, "GP")
        elif self.use_gp == 2:
            if self.gp_from_file:
                gpe = GPEnsemble.fromdir(self.ensemble_path, "RGP")
            else:
                X_basis = [np.linspace(-self.v_max, self.v_max, self.n_basis_vectors) for _ in range(3)]
                gpe = GPEnsemble.fromemptybasisvectors(X_basis)
        else:
            raise ValueError("Invalid GPE argument")

        # Creates an optimizer object for the quad
        self.quad_opt = quad_optimizer(quad, t_horizon=self.t_lookahead, n_nodes=self.n_nodes, gpe=gpe) # computing optimal control over model of plant

        # MPC steps at a different rate than the odometry
        # Trajectory steps at odometry rate
        self.control_freq_factor = int(self.quad_opt.optimization_dt / self.ODOMETRY_DT) # MPC takes trajectory steps as input -> I need to correct these steps to the MPC rate





    def pose_received_cb(self, msg : Odometry):
        """
        Callback function for pose subscriber. When new odometry message is received, the controller is used to calculate new inputs.
        Publishes calculated inputs to the autopilot
        :param msg: Odometry message of type nav_msgs/Odometry
        """
        #!IMPORTANT: This function runs FIFO, not from the most recent message.

        # I ignore odometry unless I have a trajectory. This is to avoid the controller to start before the trajectory is received
        # Trajectory is received in the trajectory_received_cb function
        # New trajectory is requested elsewhere
        x, timestamp_odometry = self.pose_to_state_world(msg)



        if timestamp_odometry < self.last_reboot_timestamp:
            # Dump the accumulated odometry messages that came before the reboot was finished
            #rospy.logwarn("Dumping odometry messages")
            #rospy.logwarn(f"{timestamp_odometry} < {self.last_reboot_timestamp}")
            return

        time_at_cb_start = time.time()

        if self.need_trajectory_to_hover:
            # The controller just started and I am waiting for the first trajectory
            self.need_trajectory_to_hover = False
            self.trajectory_ready = False
            #sx, _ = self.pose_to_state_world(msg)
            
            if np.linalg.norm(x[0:3] - self.HOVER_POS) > self.EPSILON_TRAJECTORY_FINISHED:
                # I am not at the hover height, so I need to request a trajectory to the hover height
                start_pos = np.array([x[0], x[1], x[2]])
                

                # Used to skip counting this line trajectory as a finished trajectory, since its only for initial alignment
                self.doing_a_line = True
                # Take me from here to the hover position
                self.publish_trajectory_request("line", start_pos, self.HOVER_POS, v_max=self.v_max, a_max=self.a_max)
            else:
                # I am at the hover height, so I can request a new trajectory
                self.request_trajectory(x, self.trajectory_type)



        if not self.need_trajectory_to_hover and self.trajectory_ready:
               
                # Get current state from gazebo
                
                self.quad_opt.set_quad_state(x) # This line is superfluous I think.

                # Reference is sampled with 100 Hz, but mpc step is 1s/10 = 0.1s
                # That means I need to take only every 10th reference point # control freq factor
                x_ref = get_reference_chunk(self.x_trajectory, self.idx_traj, self.quad_opt.n_nodes, self.control_freq_factor)
                t_ref = get_reference_chunk(self.t_trajectory, self.idx_traj, self.quad_opt.n_nodes, self.control_freq_factor)
                self.quad_opt.set_reference_trajectory(x_ref)
                
                # -------------- Solve the optimization problem --------------
                time_before_mpc = time.time()
                x_opt, w_opt, t_cpu, cost_solution = self.quad_opt.run_optimization(x)
                elapsed_during_mpc = time.time() - time_before_mpc
                #rospy.loginfo(f"Elapsed time during MPC: \n\r {elapsed_during_mpc*1000:.3f} ms")
                
                # MPC uses only the first control command
                w = w_opt[0, :]
                # Last three elements of x_opt are the body rates
                self.send_control_command(w, x_opt[1,10:13])

                self.idx_traj += 1

                # -------------- RGP regress --------------
                if self.quad_opt.gpe.type == 'RGP':
                    rgp_basis_vectors = [self.quad_opt.gpe.gp[d].X
                            for d in range(len(self.quad_opt.gpe.gp))]
                    if self.logger.dictionary:
                        x_pred_minus_1 = self.logger.dictionary['x_pred_odom'][-1]
                    else:
                        x_pred_minus_1 = x

                    rgp_mu_g_t, rgp_C_g_t = self.quad_opt.regress_and_update_RGP_model(x, x_pred_minus_1)
                    rgp_theta = self.quad_opt.gpe.get_theta()
                else:
                    # If not using RGP, set these to None for logging
                    rgp_basis_vectors = None
                    rgp_mu_g_t = None
                    rgp_C_g_t = None
                    rgp_theta = None

                # ------- Publish visualisations to rviz -------

                # Part of the current trajectory that is used for the optimization for control
                reference_chunk_path = self.trajectory_chunk_to_path(x_ref, t_ref)
                self.reference_path_chunk_pub.publish(reference_chunk_path)
                # The path found by the optimization for control
                # Add one more dt to the end of t_ref because the MPC is solving for n=0, ..., N and t_ref is for n=0, ..., N-1
                optimal_path = self.trajectory_chunk_to_path(x_opt[:,:], np.concatenate((t_ref[-1] + t_ref[-1]-t_ref[-2], t_ref.reshape(-1))))

                self.publish_marker_to_rviz(x_ref[0,0:3])
                self.optimal_path_pub.publish(optimal_path)

                # Predict the state at the next odometry message for logging purposes
                x_pred = self.quad_opt.discrete_dynamics(x, w, self.ODOMETRY_DT)
                #x_pred = x_opt[1,:]

                # ------- Log data -------
                
                if self.logger is not None:
                    dict_to_log = {"x_odom": x, "x_pred_odom": x_pred, "x_ref": x_ref[0,:], "t_odom": timestamp_odometry, \
                        "w_odom": w, 't_cpu': t_cpu, "elapsed_during_mpc": elapsed_during_mpc, "cost_solution": cost_solution, \
                            "rgp_basis_vectors" : rgp_basis_vectors, "rgp_mu_g_t": rgp_mu_g_t, "rgp_C_g_t": rgp_C_g_t, "rgp_theta": rgp_theta}
                    
                    self.logger.log(dict_to_log)

                

                # tqdm progress bar update. Only displays when this script is ran as main. Not while using roslaunch
                if self.pbar is not None and self.idx_traj < self.t_trajectory.shape[0]:
                    self.pbar.update(1)

                # -------------- Check if the trajectory is finished --------------
                
                if self.idx_traj+1 == self.x_trajectory.shape[0] and np.linalg.norm(x[0:3] - x_ref[0,0:3]) < self.EPSILON_TRAJECTORY_FINISHED:
                    
                    # The trajectory is finished
                    rospy.loginfo("Trajectory finished")

                    if self.number_of_trajectories_finished >= self.training_trajectories_count:
                        # Shutdown the node after making the required number of trajectories
                        #rospy.on_shutdown(self.shutdown_hook) # Send the signal that this process is about to shutdown
                        rospy.logwarn("Data collection finished.")
                        #sys.exit("Number of trajectories finished") # End this python process
                    else:
                        # Not finished yet, request a new trajectory

                        if self.doing_a_line:
                            # This was a line trajectory, dont count it as a finished trajectory
                            self.logger.clear_memory()
                            self.doing_a_line = False
                            self.request_trajectory(x, self.trajectory_type)
                        else:
                            # This was a real trajectory, count it as a finished trajectory
                            self.number_of_trajectories_finished += 1
                            self.logger.save_log() # Saves the log to a file

                            if not self.training_run:
                                # Clear memory after every trajectory when not collecting data for training
                                self.logger.clear_memory()
                                rospy.loginfo("Cleared logger memory because this is not a training run")
                            rospy.loginfo(f"Explore: {self.explore}")


                            if self.explore:

                                self.rebooting_controller = True

                                # Let autopilot take over
                                self.x_hover = x
                                self.send_go_to_pose_autopilot_command(self.x_hover)
                                self._force_hover_pub.publish(Empty())
                                #rospy.logwarn("Sent go to pose command to autopilot")
                                rospy.logwarn("Retraining controller")
                                # Explore the state space
                                self.retrain_controller()
                                rospy.logwarn("Retrained controller with new gp")
                                
                                #self.mpc_ros_wrapper.quad_opt.acados_ocp.model.


                                # Decide what velocity to use for the next trajectory
                                self.explorer = Explorer(self.quad_opt.gpe)
                                self.v_max = self.explorer.velocity_to_explore
                                self.a_max = self.explorer.velocity_to_explore

                                rospy.logwarn(f"Exploring with velocity {self.v_max} m/s")
                                #rospy.logwarn(f"Wrapper quad opt gp: {self.mpc_ros_wrapper.quad_opt.gpe}")

                                # Controller rebooted, now we can go as normal
                                self.rebooting_controller = False
                                self.rebooted_controller = True
                                # Remember the time when the controller was rebooted
                                # This is used to dump callbacks that accumulate during the reboot
                                self.last_reboot_timestamp = rospy_time_to_float(rospy.Time.now())
                                rospy.logwarn(f"Last reboot time: {self.last_reboot_timestamp}")

                            self.request_trajectory(x, self.trajectory_type)


        elapsed_during_cb = time.time() - time_at_cb_start
        #rospy.loginfo(f"Elapsed time during callback: \n\r {elapsed_during_cb*1000:.3f} ms")


    def retrain_controller(self):
        """ 
        # TODO: ! DOES NOT WORK !
        Trains the controller with the data collected so far. Then reinitializes the quad_optimizer inside the MPC wrapper with the new GPE. 
        """ 
        # Now I definitely have access to a gp
        self.use_gp = True
        self.use_gp = True
        dir_path = os.path.dirname(os.path.realpath(__file__))

        gpefit_plot_filepath = os.path.join(dir_path, '..', 'outputs', 'graphics', 'gpefit_' + "retrain" + '.pdf')
        gpesamples_plot_filepath = os.path.join(dir_path, '..', 'outputs', 'graphics', 'gpesamples_' + "retrain" + '.pdf')
        train_gp(self.logger.filepath_dict, self.ensemble_path, n_training_samples=10, theta0=None, show_plots=False, gpefit_plot_filepath=gpefit_plot_filepath, gpesamples_plot_filepath=gpesamples_plot_filepath)

        # Reinitialize MPC solver with the retrained GPE
        
        #self.mpc_ros_wrapper.initialize()



    def request_trajectory(self, x : np.ndarray, trajectory_type : str):           
        """
        Wrapper for publish_trajectory_request which uses preset endpoints and velocities.
        :param x: Current state of the quadrotor that is possibly used in the trajectory request
        :param trajectory_type: Type of trajectory to request
        """
        if trajectory_type == "static":
            self.publish_trajectory_request("static", \
                start_point=None, end_point=None, \
                    v_max=self.v_max, a_max=self.a_max)
        
        if trajectory_type == "random":
            self.publish_trajectory_request("random", \
                start_point=np.array([x[0], x[1], x[2]]), end_point=None, \
                    v_max=self.v_max, a_max=self.a_max)

        
        if trajectory_type == "circle":
            radius = 10.0
            end_point = np.array([x[0]+radius, x[1], x[2]]) # Circle trajectory radius is calculated as the distance between start and end
            self.publish_trajectory_request("circle", \
                start_point=np.array([x[0], x[1], x[2]]), end_point=end_point, \
                    v_max=self.v_max, a_max=self.a_max)


    def publish_trajectory_request(self, type : str, start_point : np.ndarray = np.array([0, 0, 0]), end_point : np.ndarray = np.array([0, 0, 0]), v_max : float = None, a_max : float = None):
        """
        Publishes a trajectory request of type Trajectory_request to the trajectory generator node.
        :param: type: Type of the trajectory that is requested. Can be "static", "random", "circle" or "line"
        :param: start_point: Start point of the trajectory. If None, send its absence in the message
        :param: end_point: End point of the trajectory. If None, send its absence in the message
        :param: v_max: Maximum velocity of the trajectory. If None, uses self.v_max
        :param: a_max: Maximum acceleration of the trajectory. If None, uses self.a_max
        """
        r = rospy.Rate(1)
        msg = Trajectory_request()
        msg.type = String(type)
        
        # -------- Start and end point handling --------
        if start_point is None:
            # If no end point is given, send that information in the message. end_point has to be instantiated in the message. 
            # Dont want to look up if I can maybe fill with NaNs
            msg.start_point_enabled = Bool(False)
            msg.start_point = Point(0.0, 0.0, 0.0)
        else:
            msg.start_point_enabled = Bool(True)
            msg.start_point = Point(start_point[0], start_point[1], start_point[2])
        
        if end_point is None:
            # If no end point is given, send that information in the message. end_point has to be instantiated in the message. 
            # Dont want to look up if I can maybe fill with NaNs
            msg.end_point_enabled = Bool(False)
            msg.end_point = Point(0.0, 0.0, 0.0)
        else:
            msg.end_point_enabled = Bool(True)
            msg.end_point = Point(end_point[0], end_point[1], end_point[2])
        
        # -------- Velocity and acceleration handling --------
        if v_max is None:
            msg.v_max = Float32(self.v_max)
        else:
            msg.v_max = Float32(v_max)

        if a_max is None:
            msg.a_max = Float32(self.a_max)
        else:
            msg.a_max = Float32(a_max)

        self.new_trajectory_request_pub.publish(msg)
        
        rospy.loginfo(f"Requested new trajectory: \n\r type: {type} \n\r  {start_point} --> {end_point} \n\r v_max={v_max}, a_max={a_max} \n\r at topic {self.new_trajectory_request_topic}")
        
        


    def trajectory_received_cb(self, msg : Trajectory):
        """
        Callback function for trajectory subscriber. New path is received. Reset idx_traj to 0 and set x_trajectory to new path for following.
        :param: msg: Trajectory message
        """
        
        samples_in_trajectory = len(msg.timeStamps)
        rospy.loginfo(f"Parsing new trajectory with {samples_in_trajectory} samples")
        self.idx_traj = 0

        self.x_trajectory = np.empty((samples_in_trajectory, 13)) # x, y, z, w, x, y, z, vx, vy, vz, wx, wy, wz
        self.t_trajectory = np.empty((samples_in_trajectory, 1)) # t

        # -------- Parse message --------
        for i in range(samples_in_trajectory):
            self.t_trajectory[i] = msg.timeStamps[i].stamp.secs + msg.timeStamps[i].stamp.nsecs * 1e-9 # Convert to seconds

            self.x_trajectory[i, 0] = msg.positions[i].x
            self.x_trajectory[i, 1] = msg.positions[i].y
            self.x_trajectory[i, 2] = msg.positions[i].z

            self.x_trajectory[i, 3] = msg.orientations[i].w
            self.x_trajectory[i, 4] = msg.orientations[i].x
            self.x_trajectory[i, 5] = msg.orientations[i].y
            self.x_trajectory[i, 6] = msg.orientations[i].z

            self.x_trajectory[i, 7] = msg.velocities[i].x
            self.x_trajectory[i, 8] = msg.velocities[i].y
            self.x_trajectory[i, 9] = msg.velocities[i].z

            self.x_trajectory[i, 10] = msg.rates[i].x
            self.x_trajectory[i, 11] = msg.rates[i].y
            self.x_trajectory[i, 12] = msg.rates[i].z


        self.trajectory_ready = True
        self.wait_for_trajectory = False
        self.pbar = tqdm(total=samples_in_trajectory) 
        rospy.logwarn("Received new trajectory with duration {}s".format(self.t_trajectory[-1] - self.t_trajectory[0]))
        
        

    def send_go_to_pose_autopilot_command(self, x : np.ndarray):
        """
        Send a go to pose command to the autopilot
        :param: x: 13D vector. Sends the position, and simple orientation (yaw) in plane to the autopilot
        """
        go_to_pose_msg = PoseStamped()
        go_to_pose_msg.pose.position.x = float(x[0])
        go_to_pose_msg.pose.position.y = float(x[1])
        go_to_pose_msg.pose.position.z = float(x[2])

        roll, pitch, yaw = quaternion_to_euler([x[3], x[4], x[5], x[6]])
        heading = yaw

        go_to_pose_msg.pose.orientation.w = np.cos(heading / 2.0)
        go_to_pose_msg.pose.orientation.z = np.sin(heading / 2.0)

        self._go_to_pose_pub.publish(go_to_pose_msg)

    def send_control_command_hover(self):
        """
        Relinquish control to the autopilot
        """
        # Control input command to the autopilot
        control_msg = ControlCommand()
        control_msg.header = std_msgs.msg.Header()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.control_mode = 0
        control_msg.armed = True

        self.actuator_publisher.publish(control_msg)

    def send_control_command(self, thrust : np.ndarray, body_rates : np.ndarray):
        """
        Creates a ControlCommand message and sends it to the autopilot
        :param thrust: 4x1 array with the thrust for each rotors in range [0-1]
        :param body_rates: 3x1 array with the body rates in rad/s. The autopilot needs both thrust and bodyrate
        """            
        # Control input command to the autopilot
        control_msg = ControlCommand()
        control_msg.header = std_msgs.msg.Header()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.control_mode = 2
        control_msg.armed = True

        # Autopilot needs desired body rates to set rotor speeds
        control_msg.bodyrates.x = body_rates[0]
        control_msg.bodyrates.y = body_rates[1]
        control_msg.bodyrates.z = body_rates[2]

        # Autopilot needs desired thrust to set rotor speeds
        control_msg.rotor_thrusts = thrust * self.quad_opt.quad.max_thrust / self.quad_opt.quad.mass
        #np.sum(w_opt[:4]) * self.quad.max_thrust / self.quad.mass
        control_msg.collective_thrust = np.sum(thrust) * self.quad_opt.quad.max_thrust / self.quad_opt.quad.mass

        self.actuator_publisher.publish(control_msg)

    def trajectory_chunk_to_path(self, x_ref : np.ndarray, t_ref : np.ndarray):
        """
        Converts a trajectory chunk to a ROS Path message for visualization
        :param x_ref: 13D array with the trajectory
        :param t_ref: 1D array with the time stamps
        :return: ROS Path message
        """

        path = Path()
        path.poses = [PoseStamped()]*len(t_ref)
        path.header.frame_id = "world"
        #rospy.loginfo(f'len(path.poses): {len(path.poses)}')
        for i in range(t_ref.shape[0]):
            pose_stamped = PoseStamped()

            # Pose at time t
            pose_stamped.pose.position.x = x_ref[i, 0]
            pose_stamped.pose.position.y = x_ref[i, 1]
            pose_stamped.pose.position.z = x_ref[i, 2]


            # Referential frame of the pose
            pose_stamped.header.frame_id = "world"

            # Time t
            # Convert seconds to the required stamp format
            seconds = int(t_ref[i])
            nanoseconds = int(int(t_ref[i] * 1e9) - seconds * 1e9) # Integer arithmetic is strange
            #rospy.loginfo(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
            pose_stamped.header.stamp.secs = seconds
            pose_stamped.header.stamp.nsecs = nanoseconds
            
            path.poses[i] = pose_stamped
        return path
        


    def pose_to_state(self, msg : Odometry) -> Tuple[np.ndarray, float]:
        """
        Convert pose message to state vector with velocity in body frame
        :param msg: ROS Odometry message
        :return: (state, timestamp)
        """
        p = msg.pose.pose.position # position # 3 x float64
        q = msg.pose.pose.orientation # quaternion # 4 x float64
        v = msg.twist.twist.linear # linear velocity # 3 x float64
        r = msg.twist.twist.angular # angular velocity # 3 x float64

        timestamp = rospy_time_to_float(msg.header.stamp)

        state = np.array([p.x, p.y, p.z, q.w, q.x, q.y, q.z, v.x, v.y, v.z, r.x, r.y, r.z])
        return state, timestamp
        
    def pose_to_state_world(self, msg : Odometry) -> Tuple[np.ndarray, float]:
        '''
        Convert pose message to state vector in world frame
        :param msg: ROS Odometry message
        :return: (state, timestamp)
        '''
        x, timestamp = self.pose_to_state(msg)
        q = x[3:7]
        v = x[7:10]
        v_world = v_dot_q(v, q)
        return np.array([x[0], x[1], x[2], q[0], q[1], q[2], q[3], v_world[0], v_world[1], v_world[2], x[10], x[11], x[12]]), timestamp


    def publish_marker_to_rviz(self, p : np.ndarray):
        """
        Publish a marker to rviz for visualization
        :param p: reference position on the trajectory
        """
        robotMarker = Marker()
        #robotMarker.header.frame_id = "hummingbird/base_link"
        robotMarker.header.frame_id = "world"
        robotMarker.header.stamp    = rospy.get_rostime()
        robotMarker.ns = "hummingbird"
        robotMarker.id = 0
        robotMarker.type = 2 # sphere
        robotMarker.action = 0
        robotMarker.pose.position.x = p[0]
        robotMarker.pose.position.y = p[1]
        robotMarker.pose.position.z = p[2]
        robotMarker.pose.orientation.x = 0
        robotMarker.pose.orientation.y = 0
        robotMarker.pose.orientation.z = 0
        robotMarker.pose.orientation.w = 1.0
        robotMarker.scale.x = 0.1
        robotMarker.scale.y = 0.1
        robotMarker.scale.z = 0.1

        robotMarker.color.r = 0.0
        robotMarker.color.g = 1.0
        robotMarker.color.b = 0.0
        robotMarker.color.a = 1.0
        robotMarker.lifetime = rospy.Duration(0) # forever
        self.markerPub.publish(robotMarker)

    
    def shutdown_hook(self):
        rospy.loginfo("Shutting down MPC node")


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    controller = MPC_controller()
    rospy.spin()