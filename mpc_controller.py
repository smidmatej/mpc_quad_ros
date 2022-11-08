#!/usr/bin/env python3
import rospy


from geometry_msgs.msg import Pose
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker
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

        marker_topic = "/visualization_marker"
        self.markerPub = rospy.Publisher(marker_topic, Marker, queue_size=10)

        self.mpc_ros_wrapper = MPCROSWrapper(quad_name=quad_name)


        
        # trajectory has a specific time step that I do not respect here
        #self.x_trajectory, t_trajectory = load_trajectory("source/trajectory_generation/trajectories/trajectory_sampled.csv")


        odometry_dt = 1/100
        print(f'odometry_dt: {odometry_dt}')
        waypoint_filename = 'source/trajectory_generation/waypoints/static_waypoints.csv'
        output_trajectory_filename = 'source/trajectory_generation/trajectories/trajectory_sampled.csv'
        # Create trajectory from waypoints. One sample for every odometry message.   
        create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max=10, a_max=10, dt=odometry_dt) # Odometry is published by the simulator at 100 Hz!
        self.x_trajectory, self.t_trajectory = load_trajectory(output_trajectory_filename)


        print(f' x_trajectory.shape: ', self.x_trajectory.shape)
        #self.x_trajectory, t_trajectory = load_trajectory("source/trajectory_generation/trajectories/trajectory_sampled.csv")

        #yref, yref_N = self.quad_opt.set_reference_trajectory(x_trajectory)
        #xf = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #self.mpc_ros_wrapper.quad_opt.set_reference_state(x_target=xf)
        
        self.idx_traj = 0

        self.pose_subscriber = rospy.Subscriber(pose_topic, Odometry, self.pose_received_cb) # Pose is published by the simulator at 100 Hz!
        self.actuator_publisher = rospy.Publisher(control_topic, ControlCommand, queue_size=1, tcp_nodelay=True)


    def pose_received_cb(self, msg):


        #xf = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #self.mpc_ros_wrapper.quad_opt.set_reference_state(x_target=xf)

        print(f' x_trajectory.shape: ', self.x_trajectory.shape)
        x = self.pose_to_state_world(msg)
        self.mpc_ros_wrapper.quad_opt.set_quad_state(x)
        print(f'p: {x[0:3]}')
        print(f'q: {x[3:7]}')

        x_ref = get_reference_chunk(self.x_trajectory, self.idx_traj, self.mpc_ros_wrapper.quad_opt.n_nodes)

        self.publish_marker_to_rviz(x_ref[0,0:3])
        yref, yref_N = self.mpc_ros_wrapper.quad_opt.set_reference_trajectory(x_ref)
        self.idx_traj += 1
        

        x_opt, w_opt, t_cpu, cost_solution = self.mpc_ros_wrapper.quad_opt.run_optimization(x)
        #print(f'w_opt: {w_opt}')
        w = w_opt[0, :]
        #print(f'w_opt.shape: {w_opt.shape}')
        print(f'w: {w}')

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