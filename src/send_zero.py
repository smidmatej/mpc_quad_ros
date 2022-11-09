#!/usr/bin/env python3
import rospy

from mav_msgs.msg import Actuators


        
if __name__ == '__main__':


    rospy.init_node("Hummingbird_Controller")
    quad_name = 'hummingbird'
    control_topic  = quad_name + "/command/motor_speed"
    actuator_publisher = rospy.Publisher(control_topic, Actuators, queue_size=1, tcp_nodelay=True)
    control_msg = Actuators()
    u = 100
    control_msg.angular_velocities = [u,u,u,u]
    print("control: {}".format(control_msg.angular_velocities))

    actuator_publisher.publish(control_msg)



