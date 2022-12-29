#!/usr/bin/env python3
import rospy

# For sending and receiving trajectory messages
from geometry_msgs.msg import Point, Vector3, Quaternion
from mpcros.msg import Trajectory, Trajectory_request
from std_msgs.msg import Header


# for path visualization
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped 
from visualization_msgs.msg import Marker

import warnings


import numpy as np
import os 

from utils.utils import get_reference_chunk, v_dot_q, get_reference_chunk
from trajectory_generation.generate_trajectory import write_waypoints_to_file, generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

from trajectory_generation.TrajectoryGenerator import TrajectoryGenerator

class TrajectoryBuilder:
        
    def __init__(self):
        
        self.trajectory_generator = TrajectoryGenerator()

        quad_name = 'hummingbird'

        rospy.init_node("trajectory_builder")

        # TODO: switch all the rospy.loginfo statements to rospy.loginfo
        # Also find out some best practices for logging in ROS


        # Odometry is published with a average frequency of 100 Hz
        # TODO: check if this is the case, sometimes it is delayed a bit
        self.trajectory_dt = 1/100


        #execution_path = os.path.dirname(os.path.realpath(__file__))
        this_path = os.path.dirname(os.path.realpath(__file__))
        rospy.loginfo(f'this_path: {this_path}')
        # For user defined waypoints
        self.static_waypoint_filename = 'trajectory_generation/waypoints/static_waypoints.csv'
        # For generating trajectory between two points
        self.line_waypoint_filename = 'trajectory_generation/waypoints/static_waypoints.csv'
        # For generating random waypoints for a trajectory
        self.random_waypoint_filename = 'trajectory_generation/waypoints/random_waypoints.csv'
        # The final trajectory is sampled and saved here
        self.output_trajectory_filename = 'trajectory_generation/trajectories/trajectory_sampled.csv'

        # Topics
        self.trajectory_topic = "reference/trajectory"
        self.new_trajectory_request_topic = "reference/new_trajectory_request"

        self.marker_topic = "rviz/marker"
        self.path_rviz_topic = "rviz/path" 
        

        
        self.new_trajectory_request_sub = rospy.Subscriber(self.new_trajectory_request_topic, Trajectory_request, self.new_trajectory_request_cb)

        self.markerPub = rospy.Publisher(self.marker_topic, Marker, queue_size=10)

        self.path_rviz_Pub = rospy.Publisher(self.path_rviz_topic, Path, queue_size=10) # rviz vizualization of the trajectory
        self.trajectoryPub = rospy.Publisher(self.trajectory_topic, Trajectory, queue_size=1) # trajectory to be used by the controller
 



        #self.set_new_trajectory(type='static')

        rospy.loginfo("Trajectory publisher initialized")

 



    def new_trajectory_request_cb(self, msg):

        type = msg.type.data
        
        if msg.start_point_enabled.data == True:
            start_point = np.array([msg.start_point.x, msg.start_point.y, msg.start_point.z])
        else:
            start_point = None
        
        # Complicated way to pass a None object through ROS
        if msg.end_point_enabled.data == True:
            end_point = np.array([msg.end_point.x, msg.end_point.y, msg.end_point.z])
        else:
            end_point = None

        v_max = msg.v_max.data
        a_max = msg.a_max.data

        '''
        rospy.loginfo("New trajectory requested")
        rospy.loginfo(f'Type: {type}')
        rospy.loginfo(f'Start point: \n\r {start_point}')
        rospy.loginfo(f'End point: \n\r {end_point}')
        rospy.loginfo(f'end_point_enabled: {msg.end_point_enabled.data}')
        rospy.loginfo(f'v_max: {v_max}')
        rospy.loginfo(f'a_max: {a_max}')
        '''

        rospy.loginfo(f"Received trajectory request: \n\r type: {type} \n\r  {start_point} --> {end_point} \n\r v_max={v_max}, a_max={a_max} \n\r at topic {self.new_trajectory_request_topic}")
        

        self.set_new_trajectory(type, start_point, end_point, v_max, a_max)
        self.publish_trajectory()
        self.publish_trajectory_to_rviz()

        rospy.loginfo(f"Published trajectory to {self.path_rviz_topic} and to {self.trajectory_topic}")


    def set_new_trajectory(self, type='circle', start_point=np.array([0,0,0]), end_point=np.array([0,0,0]), v_max=1.0, a_max=1.0):
        """
        Generates a new trajectory and saves it to self.x_trajectory and self.t_trajectory based on the type of trajectory
        :param type: type of trajectory to generate (circle, random, static)
        """


        if type == 'line':
            rospy.loginfo('Generating line trajectory')
            assert start_point is not None and end_point is not None, "start_point and end_point should not be None for line between two points"
            
            
            self.trajectory_generator.write_waypoints_to_file([start_point, end_point])
            # Create trajectory from waypoints with the same dt as the MPC control frequency    
            self.trajectory_generator.sample_trajectory(type, v_max, a_max, self.trajectory_dt)

        if type == 'static':
            assert start_point is None and end_point is None, "start_point and end_point should be None for static trajectory"
            # Create trajectory from waypoints with the same dt as the MPC control frequency    
            self.trajectory_generator.sample_trajectory(type, v_max, a_max, self.trajectory_dt)

        
        if type == 'random':
            # Generate trajectory as reference for the quadrotor
            # new trajectory
            hsize = [10, 10, 10]
            num_waypoints = 10
            # First generate random waypoints
            self.trajectory_generator.generate_random_waypoints(hsize=hsize, num_waypoints=num_waypoints, start_point=start_point, end_point=end_point)
            # Then interpolate the waypoints to create a trajectory
            self.trajectory_generator.sample_trajectory(type, v_max, a_max, self.trajectory_dt)



        if type == 'circle':
            # Circle trajectory
            t_max = 30.0

            # Circle trajectory has no endpoint, but we pass the endpoint in to message because we compute the 
            # radius as dist(start_point, end_point)
            assert start_point is not None and end_point is not None, "start_point and end_point should not be None for circle trajectory"
            radius = np.linalg.norm(start_point - end_point)

            #assert end_point is None, "End point should be None for circle trajectory, because we dont know the end"
            self.trajectory_generator.sample_circle_trajectory_accelerating(radius, v_max, t_max=t_max, dt=self.trajectory_dt, start_point=start_point)
            

        # Loads the sampled trajectory from file to self.x_trajectory and self.t_trajectory
        self.x_trajectory, self.t_trajectory = self.trajectory_generator.load_trajectory()

    def publish_trajectory(self):
        """
        Creates a custom trajectory message with the trajectory and publishes it to the trajectory topic
        Trajectory is basically the contents of the trajectory_sampled file
        CAREFUL: The trajectory only contains position, velocity and acceleration information. The orientation and rates are made up
        """
        traj = Trajectory()
        

        traj.timeStamps = [None]*len(self.t_trajectory)
        traj.positions = [Point()]*len(self.t_trajectory)
        traj.orientations = [Quaternion()]*len(self.t_trajectory)
        traj.velocities = [Vector3()]*len(self.t_trajectory)
        traj.rates = [Vector3()]*len(self.t_trajectory)


        traj.header.frame_id = "world"
        #rospy.loginfo(f'len(path.poses): {len(path.poses)}')
        for i in range(self.t_trajectory.shape[0]):
            

            traj.positions[i] = Point(self.x_trajectory[i, 0], self.x_trajectory[i, 1], self.x_trajectory[i, 2])

            # w property of the quaternion is last in the object but first in x_trajectory
            traj.orientations[i] = Quaternion(self.x_trajectory[i, 4], self.x_trajectory[i, 5], self.x_trajectory[i, 6], self.x_trajectory[i, 3])
            traj.velocities[i] = Vector3(self.x_trajectory[i, 7], self.x_trajectory[i, 8], self.x_trajectory[i, 9])
            traj.rates[i] = Vector3(self.x_trajectory[i, 10], self.x_trajectory[i, 11], self.x_trajectory[i, 12])





            # Referential frame of the pose
            traj.timeStamps[i] = Header()
            traj.timeStamps[i].frame_id = "world"

            # Time t
            # Convert seconds to the required stamp format
            seconds = int(self.t_trajectory[i])
            nanoseconds = int(int(self.t_trajectory[i] * 1e9) - seconds * 1e9) # Integer arithmetic is strange
            #rospy.loginfo(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
            traj.timeStamps[i].stamp.secs = seconds
            traj.timeStamps[i].stamp.nsecs = nanoseconds


        self.trajectoryPub.publish(traj)


    def publish_trajectory_to_rviz(self):

        path = Path()
        path.poses = [PoseStamped()]*len(self.t_trajectory)
        path.header.frame_id = "world"
        #rospy.loginfo(f'len(path.poses): {len(path.poses)}')
        for i in range(self.t_trajectory.shape[0]):
            pose_stamped = PoseStamped()

            # Pose at time t
            pose_stamped.pose.position.x = self.x_trajectory[i, 0]
            pose_stamped.pose.position.y = self.x_trajectory[i, 1]
            pose_stamped.pose.position.z = self.x_trajectory[i, 2]


            # Referential frame of the pose
            pose_stamped.header.frame_id = "world"

            # Time t
            # Convert seconds to the required stamp format
            seconds = int(self.t_trajectory[i])
            nanoseconds = int(int(self.t_trajectory[i] * 1e9) - seconds * 1e9) # Integer arithmetic is strange
            #rospy.loginfo(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
            pose_stamped.header.stamp.secs = seconds
            pose_stamped.header.stamp.nsecs = nanoseconds
            
            path.poses[i] = pose_stamped

        self.path_rviz_Pub.publish(path)
        

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
    traj = Trajectory()
    np.set_printoptions(precision=2)
    traj_builder = TrajectoryBuilder()
    
    
    rospy.spin()
