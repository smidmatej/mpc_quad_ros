#!/usr/bin/env python3
import rospy


from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import Header
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

class TrajecroryBuilder:
        
    def __init__(self):

        rospy.init_node("TrajecroryBuilder", anonymous=True)

        quad_name = 'hummingbird'


        marker_topic = quad_name + "/rviz/marker"
        path_rviz_topic = quad_name + "/rviz/path" 
        trajectory_topic = quad_name + "/reference/trajectory"
        self.markerPub = rospy.Publisher(marker_topic, Marker, queue_size=10)

        self.path_rviz_Pub = rospy.Publisher(path_rviz_topic, Path, queue_size=1) # rviz vizualization of the trajectory
        self.trajectoryPub = rospy.Publisher(trajectory_topic, Trajectory, queue_size=1) # trajectory to be used by the controller
 
        # Odometry is published with a average frequency of 100 Hz
        # TODO: check if this is the case, sometimes it is delayed a bit
        self.trajectory_dt = 1/100

        execution_path = os.path.dirname(os.path.realpath(__file__))
        self.waypoint_filename = execution_path + '/source/trajectory_generation/waypoints/static_waypoints.csv'
        self.output_trajectory_filename = execution_path + '/source/trajectory_generation/trajectories/trajectory_sampled.csv'

        self.set_new_trajectory()

        print("Trajectory publisher initialized")

    def set_new_trajectory(self):
        # Create trajectory from waypoints. One sample for every odometry message.   
        create_trajectory_from_waypoints(self.waypoint_filename, self.output_trajectory_filename, v_max=5, a_max=5, dt=self.trajectory_dt) # Odometry is published by the simulator at 100 Hz!
        
        
        self.x_trajectory, self.t_trajectory = load_trajectory(self.output_trajectory_filename)

        self.publish_trajectory(self.x_trajectory, self.t_trajectory)
        self.publish_trajectory_to_rviz(self.x_trajectory, self.t_trajectory)


    def publish_trajectory(self, x_trajectory, t_trajectory):

        traj = Trajectory()
        traj.timeStamps = [None]*len(t_trajectory)
        traj.positions = [Point()]*len(t_trajectory)
        traj.velocities = [Vector3()]*len(t_trajectory)
        traj.accelerations = [Vector3()]*len(t_trajectory)

        traj.header.frame_id = "world"
        #print(f'len(path.poses): {len(path.poses)}')
        for i in range(t_trajectory.shape[0]):



            # Pose at time t
            traj.positions[i].x = x_trajectory[i, 0]
            traj.positions[i].y = x_trajectory[i, 1]
            traj.positions[i].z = x_trajectory[i, 2]

            traj.velocities[i].x = x_trajectory[i, 3]
            traj.velocities[i].y = x_trajectory[i, 4]
            traj.velocities[i].z = x_trajectory[i, 5]

            traj.accelerations[i].x = x_trajectory[i, 6]
            traj.accelerations[i].y = x_trajectory[i, 7]
            traj.accelerations[i].z = x_trajectory[i, 8]


            # Referential frame of the pose
            traj.timeStamps[i] = Header()
            traj.timeStamps[i].frame_id = "world"

            # Time t
            # Convert seconds to the required stamp format
            seconds = int(t_trajectory[i])
            nanoseconds = int(int(t_trajectory[i] * 1e9) - seconds * 1e9) # Integer arithmetic is strange
            #print(f'seconds: {seconds}, nanoseconds: {nanoseconds}')
            traj.timeStamps[i].stamp.secs = seconds
            traj.timeStamps[i].stamp.nsecs = nanoseconds


        self.trajectoryPub.publish(traj)


    def publish_trajectory_to_rviz(self, x_trajectory, t_trajectory):

        path = Path()
        path.poses = [PoseStamped()]*len(t_trajectory)
        path.header.frame_id = "world"
        #print(f'len(path.poses): {len(path.poses)}')
        for i in range(t_trajectory.shape[0]):
            pose_stamped = PoseStamped()

            # Pose at time t
            pose_stamped.pose.position.x = x_trajectory[i, 0]
            pose_stamped.pose.position.y = x_trajectory[i, 1]
            pose_stamped.pose.position.z = x_trajectory[i, 2]


            # Referential frame of the pose
            pose_stamped.header.frame_id = "world"

            # Time t
            # Convert seconds to the required stamp format
            seconds = int(t_trajectory[i])
            nanoseconds = int(int(t_trajectory[i] * 1e9) - seconds * 1e9) # Integer arithmetic is strange
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
    traj_builder = TrajecroryBuilder()

    rospy.spin()