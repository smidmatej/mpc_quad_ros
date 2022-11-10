#!/usr/bin/env python3
import rospy


from geometry_msgs.msg import Pose, Point
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32

from visualization_msgs.msg import Marker

# for path visualization
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped 


from mpcros.msg import Trajectory
from mpcros.msg import Trajectory_request

import std_msgs
import numpy as np
import os 



from MPCROSWrapper import MPCROSWrapper
from source.quad import Quadrotor3D
from source.quad_opt import quad_optimizer
from source.utils.utils import load_trajectory, get_reference_chunk, v_dot_q, get_reference_chunk
from source.trajectory_generation.generate_trajectory import generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

class MPC_controller:
        
    def __init__(self):

        rospy.init_node("Hummingbird_Controller")

        quad_name = 'hummingbird'
        pose_topic = "/" + quad_name + "/ground_truth/odometry"
        control_topic = "/" + quad_name + "/autopilot/control_command_input"

        marker_topic = quad_name + "/rviz/marker"
        reference_trajectory_topic = quad_name + "/reference/trajectory"
        reference_path_chunk_topic = quad_name + "/rviz/reference_chunk"
        optimal_path_topic = quad_name + "/rviz/optimal_path"



        # Odometry is published with a average frequency of 100 Hz
        # TODO: check if this is the case, sometimes it is delayed a bit
        self.odometry_dt = 1/100
        # TODO: Gazebo runs with a variable rate, this changes the odometry dt, but the trajectories are still sampled the same. This is clearly wrong.

        self.v_max = 5
        self.a_max = 5
         
        self.optimal_path_pub = rospy.Publisher(optimal_path_topic, Path, queue_size=1) # Path from current quad position onto the path
        self.reference_path_chunk_pub = rospy.Publisher(reference_path_chunk_topic, Path, queue_size=1) # Chunk of the reference path that is used for MPC
        self.markerPub = rospy.Publisher(marker_topic, Marker, queue_size=10)

        
        self.new_trajectory_request_topic = quad_name + "/reference/new_trajectory_request"
        self.new_trajectory_request_pub = rospy.Publisher(self.new_trajectory_request_topic, Trajectory_request, queue_size=1)

        self.trajectory_sub = rospy.Subscriber(reference_trajectory_topic, Trajectory, self.trajectory_received_cb)
     
        self.need_trajectory_to_hover = True

        # Wait a while for the subscribers to connect
        r = rospy.Rate(10)
        r.sleep()


        self.hover_pos = np.array([0, 0, 10])
        # Rise to hover height

        self.mpc_ros_wrapper = MPCROSWrapper(quad_name=quad_name)






        self.pose_subscriber = rospy.Subscriber(pose_topic, Odometry, self.pose_received_cb) # Pose is published by the simulator at 100 Hz!
        self.actuator_publisher = rospy.Publisher(control_topic, ControlCommand, queue_size=1, tcp_nodelay=True)





    def request_new_trajectory(self, type, start_point=np.array([0, 0, 0]), end_point=np.array([0, 0, 0]), v_max=1.0, a_max=1.0):
        r = rospy.Rate(1)
        

        msg = Trajectory_request()
        msg.type = String(type)
        msg.start_point = Point(start_point[0], start_point[1], start_point[2])
        msg.end_point = Point(end_point[0], end_point[1], end_point[2])
        msg.v_max = Float32(v_max)
        msg.a_max = Float32(a_max)

        self.new_trajectory_request_pub.publish(msg)
        
        print(f"Requested new trajectory: \n\r type: {type} \n\r  {start_point} --> {end_point} \n\r v_max={v_max}, a_max={a_max} \n\r at topic {self.new_trajectory_request_topic}")
        
        


    def trajectory_received_cb(self, msg):
        """
        Callback function for trajectory subscriber. New path is received. Reset idx_traj to 0 and set x_trajectory to new path for following
        """

        self.idx_traj = 0

        self.x_trajectory = np.empty((len(msg.timeStamps), 13)) # x, y, z, w, x, y, z, vx, vy, vz, wx, wy, wz
        self.t_trajectory = np.empty((len(msg.timeStamps), 1)) # t
        for i in range(0, len(msg.timeStamps)):
            self.t_trajectory[i] = msg.timeStamps[i].stamp.secs + msg.timeStamps[i].stamp.nsecs * 1e-9

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
        print("Received new trajectory with duration {}".format(self.t_trajectory[-1] - self.t_trajectory[0]))
        
        




    def pose_received_cb(self, msg):
        """
        Callback function for pose subscriber. When new odometry message is received, the controller is used to calculate new inputs.
        Publishes calculated inputs to the autopilot
        """
        # I ignore odometry unless I have a trajectory. This is to avoid the controller to start before the trajectory is received
        # Trajectory is received in the trajectory_received_cb function
        # New trajectory is requested elsewhere


        if self.need_trajectory_to_hover:
            # The controller just started and I am waiting for the first trajectory
            self.need_trajectory_to_hover = False
            x = self.pose_to_state_world(msg)
            start_pos = np.array([x[0], x[1], x[2]])
            # Take me from here to the hover position
            self.request_new_trajectory("line", start_pos, self.hover_pos, v_max=3.0, a_max=3.0)
            self.trajectory_ready = False



        if not self.need_trajectory_to_hover and self.trajectory_ready:

            x = self.pose_to_state_world(msg)
            self.mpc_ros_wrapper.quad_opt.set_quad_state(x)


            x_ref = get_reference_chunk(self.x_trajectory, self.idx_traj, self.mpc_ros_wrapper.quad_opt.n_nodes)
            t_ref = get_reference_chunk(self.t_trajectory, self.idx_traj, self.mpc_ros_wrapper.quad_opt.n_nodes)
            #print(f'self.x_trajectory: {self.x_trajectory[0:10, 0:3]}')



            self.publish_marker_to_rviz(x_ref[0,0:3])
            self.mpc_ros_wrapper.quad_opt.set_reference_trajectory(x_ref)
            
            

            x_opt, w_opt, t_cpu, cost_solution = self.mpc_ros_wrapper.quad_opt.run_optimization(x)


            #print(f'w_opt: {w_opt}')
            # Part of the current trajectory that is used for the optimization for control
            reference_chunk_path = self.trajectory_chunk_to_path(x_ref, t_ref)
            self.reference_path_chunk_pub.publish(reference_chunk_path)

            
            # The path found by the optimization for control
            # Add one more dt to the end of t_ref because the MPC is solving for n=0, ..., N and t_ref is for n=0, ..., N-1
            optimal_path = self.trajectory_chunk_to_path(x_opt[:,:], np.concatenate((t_ref[-1] + t_ref[-1]-t_ref[-2], t_ref.reshape(-1))))
            #print(f'x_opt.shape: {x_opt.shape}')
            #print(f't_ref.shape: {t_ref.shape}')
            #optimal_path = self.trajectory_chunk_to_path(x_opt, t_ref)


            self.optimal_path_pub.publish(optimal_path)


            w = w_opt[0, :]

            # Control input command to the autopilot
            self.control_msg = ControlCommand()
            self.control_msg.header = std_msgs.msg.Header()
            self.control_msg.header.stamp = rospy.Time.now()
            self.control_msg.control_mode = 2
            self.control_msg.armed = True

            # Autopilot needs desired body rates to set rotor speeds
            self.control_msg.bodyrates.x = x_opt[1, -3]
            self.control_msg.bodyrates.y = x_opt[1, -2]
            self.control_msg.bodyrates.z = x_opt[1, -1]

            # Autopilot needs desired thrust to set rotor speeds
            self.control_msg.rotor_thrusts = w * self.mpc_ros_wrapper.quad.max_thrust
            self.control_msg.collective_thrust = np.sum(w) * self.mpc_ros_wrapper.quad.max_thrust 
            
            
            #print("control: {}".format(self.control_msg.collective_thrust))
            
            self.actuator_publisher.publish(self.control_msg)
            self.idx_traj += 1

            if self.idx_traj == self.x_trajectory.shape[0]:
                
                print("Trajectory finished")
                # Give me a new random trajectory from my position and back
                self.request_new_trajectory("circle", \
                    start_point=np.array([x[0], x[1], x[2]]), end_point=np.array([x[0], x[1], x[2]]), \
                        v_max=self.v_max, a_max=self.a_max)





    def trajectory_chunk_to_path(self, x_ref, t_ref):

        path = Path()
        path.poses = [PoseStamped()]*len(t_ref)
        path.header.frame_id = "world"
        #print(f'len(path.poses): {len(path.poses)}')
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
            #print(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
            pose_stamped.header.stamp.secs = seconds
            pose_stamped.header.stamp.nsecs = nanoseconds
            
            path.poses[i] = pose_stamped
        return path
        

    def pose_to_state(self, msg):
        """
        Convert pose message to state vector with velocity in body frame
        """
        p = msg.pose.pose.position # position # 3 x float64
        q = msg.pose.pose.orientation # quaternion # 4 x float64
        v = msg.twist.twist.linear # linear velocity # 3 x float64
        r = msg.twist.twist.angular # angular velocity # 3 x float64

        state = np.array([p.x, p.y, p.z, q.w, q.x, q.y, q.z, v.x, v.y, v.z, r.x, r.y, r.z])
        return state
        
    def pose_to_state_world(self, msg):
        '''
        Convert pose message to state vector in world frame
        '''
        x = self.pose_to_state(msg)
        q = x[3:7]
        v = x[7:10]
        v_world = v_dot_q(v, q)
        return np.array([x[0], x[1], x[2], q[0], q[1], q[2], q[3], v_world[0], v_world[1], v_world[2], x[10], x[11], x[12]])

    def publish_marker_to_rviz(self, p):
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



if __name__ == '__main__':
    np.set_printoptions(precision=2)
    controller = MPC_controller()
    print("controller initialized")
    rospy.spin()