#!/usr/bin/env python3
import rospy


from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
from std_msgs.msg import Header, String
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand

from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker

# for path visualization
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped 

from mpcros.msg import Trajectory

import std_msgs
import numpy as np
import os 



from MPCROSWrapper import MPCROSWrapper
from source.quad import Quadrotor3D
from source.quad_opt import quad_optimizer
from source.utils.utils import load_trajectory, get_reference_chunk, v_dot_q, get_reference_chunk
from source.trajectory_generation.generate_trajectory import generate_random_waypoints, create_trajectory_from_waypoints, generate_circle_trajectory_accelerating

class TrajectoryBuilder:
        
    def __init__(self):

        rospy.init_node("TrajectoryBuilder", anonymous=False)

        quad_name = 'hummingbird'


        # Odometry is published with a average frequency of 100 Hz
        # TODO: check if this is the case, sometimes it is delayed a bit
        self.trajectory_dt = 1/100
        self.v_max = 3
        self.a_max = 3

        execution_path = os.path.dirname(os.path.realpath(__file__))
        self.static_waypoint_filename = execution_path + '/source/trajectory_generation/waypoints/static_waypoints.csv'
        self.random_waypoint_filename = execution_path + '/source/trajectory_generation/waypoints/random_waypoints.csv'
        self.output_trajectory_filename = execution_path + '/source/trajectory_generation/trajectories/trajectory_sampled.csv'
        self.circle_trajectory_filename = execution_path + '/source/trajectory_generation/trajectories/circle_trajectory.csv'


        self.marker_topic = quad_name + "/rviz/marker"
        self.path_rviz_topic = quad_name + "/rviz/path" 
        self.trajectory_topic = quad_name + "/reference/trajectory"

        self.new_trajectory_request_topic = quad_name + "/reference/new_trajectory_request"
        self.new_trajectory_request_sub = rospy.Subscriber(self.new_trajectory_request_topic, String, self.new_trajectory_request_cb)

        self.markerPub = rospy.Publisher(self.marker_topic, Marker, queue_size=10)

        self.path_rviz_Pub = rospy.Publisher(self.path_rviz_topic, Path, queue_size=10) # rviz vizualization of the trajectory
        self.trajectoryPub = rospy.Publisher(self.trajectory_topic, Trajectory, queue_size=1) # trajectory to be used by the controller
 



        #self.set_new_trajectory(type='static')

        print("Trajectory publisher initialized")

    def set_new_trajectory(self, type='circle'):
        """
        Generates a new trajectory and saves it to self.x_trajectory and self.t_trajectory based on the type of trajectory
        :param type: type of trajectory to generate (circle, random, static)
        """

        if type == 'static':

            # Create trajectory from waypoints with the same dt as the MPC control frequency    
            create_trajectory_from_waypoints(self.static_waypoint_filename, self.output_trajectory_filename, self.v_max, self.a_max, self.trajectory_dt)
            # trajectory has a specific time step that I do not respect here
            self.x_trajectory, self.t_trajectory = load_trajectory(self.output_trajectory_filename)
        
        if type == 'random':
            # Generate trajectory as reference for the quadrotor
            # new trajectory
            hsize = 10
            num_waypoints = 3
            generate_random_waypoints(self.random_waypoint_filename, hsize=hsize, num_waypoints=num_waypoints)
            create_trajectory_from_waypoints(self.random_waypoint_filename, self.output_trajectory_filename, self.v_max, self.a_max, self.trajectory_dt)

            # trajectory has a specific time step that I do not respect here
            self.x_trajectory, self.t_trajectory = load_trajectory(self.output_trajectory_filename)

        if type == 'circle':
            # Circle trajectory
            radius = 50
            t_max = 30

            generate_circle_trajectory_accelerating(self.circle_trajectory_filename, radius, self.v_max, t_max=t_max, dt=self.trajectory_dt)
            # trajectory has a specific time step that I do not respect here
            self.x_trajectory, self.t_trajectory = load_trajectory(self.circle_trajectory_filename)



    def new_trajectory_request_cb(self, msg):
        print("New trajectory requested")
        self.set_new_trajectory(msg.data)
        self.publish_trajectory()
        self.publish_trajectory_to_rviz()

        print(f"Published trajectory to {self.path_rviz_topic} and to {self.trajectory_topic}")

    def publish_trajectory(self):

        traj = Trajectory()
        
        traj.timeStamps = [None]*len(self.t_trajectory)
        traj.positions = [Point()]*len(self.t_trajectory)
        traj.orientations = [Quaternion()]*len(self.t_trajectory)
        traj.velocities = [Vector3()]*len(self.t_trajectory)
        traj.rates = [Vector3()]*len(self.t_trajectory)


        traj.header.frame_id = "world"
        #print(f'len(path.poses): {len(path.poses)}')
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
            #print(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
            traj.timeStamps[i].stamp.secs = seconds
            traj.timeStamps[i].stamp.nsecs = nanoseconds


        self.trajectoryPub.publish(traj)


    def publish_trajectory_to_rviz(self):

        path = Path()
        path.poses = [PoseStamped()]*len(self.t_trajectory)
        path.header.frame_id = "world"
        #print(f'len(path.poses): {len(path.poses)}')
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
            #print(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
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
