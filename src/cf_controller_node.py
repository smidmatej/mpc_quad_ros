"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

import sys

import tf
import rospy

from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray

import numpy as np

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0
VICON_UPDATE_RATE = 100.0 # Hz
VICON_INTERVAL = rospy.Duration.from_sec(1.0 / VICON_UPDATE_RATE)

class MPCController:
    def __init__(self):
        rospy.init_node("cf_controller_node")
        self.cf_name = "cf4"
        self.motorCommandPublisher = rospy.Publisher("/" + self.cf_name + "/motor_command", Int32MultiArray, queue_size=1)
        self.FullStateSubsrciber = rospy.Subscriber("/" + self.cf_name + "/full_state", Odometry, self.OdometryReceivedCallback)


    def OdometryReceivedCallback(self, msg):
        if msg.header.seq % 10 != 0:
            # Drop messages that are not a multiple of 10
            return
        if rospy.Time.now().secs - msg.header.stamp.secs > 1:
            # Drop messages that are more than 1 second old
            return


        x_world = self.OdometryToState(msg)

        z_ref = 1.0
        k = 10000.0
        u_z = k*(z_ref - x_world[2])
        #rospy.loginfo(f"z = {x_world[2]}")
        #rospy.loginfo(f"u_z = {u_z}")
        motor_power = u_z*np.array([1, 1, 1, 1])
        self.motorCommandPublish(motor_power)
        

    def OdometryToState(self, msg):
        # World frame state
        state_world = np.array([
                        msg.pose.pose.position.x, 
                        msg.pose.pose.position.y, 
                        msg.pose.pose.position.z, 
                        msg.pose.pose.orientation.w, 
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z, 
                        msg.twist.twist.linear.x, 
                        msg.twist.twist.linear.y, 
                        msg.twist.twist.linear.z, 
                        msg.twist.twist.angular.x, 
                        msg.twist.twist.angular.y, 
                        msg.twist.twist.angular.z])
        return state_world


    def motorCommandPublish(self, motor_power : np.ndarray):
        """Publish motor power to the topic where a crazyswarm server is listening and relaying to the Crazyflie"""
        msg = Int32MultiArray()
        
        msg.data = [int(motor_power[0]), int(motor_power[1]), int(motor_power[2]), int(motor_power[3])]
        rospy.loginfo(msg)
        self.motorCommandPublisher.publish(msg)


if __name__ == "__main__":
    MPCController()
    rospy.spin()