#!/usr/bin/env python3
"""
Test script to manually send a navigation goal
"""

import rospy
from geometry_msgs.msg import PoseStamped
import tf2_ros

def send_goal():
    rospy.init_node('test_goal_sender')
    
    # Wait for TF
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    rospy.sleep(2.0)
    
    # Get robot's current position
    try:
        trans = tf_buffer.lookup_transform(
            'robot1_map',
            'robot1_base_footprint',
            rospy.Time(0),
            rospy.Duration(1.0)
        )
        
        robot_x = trans.transform.translation.x
        robot_y = trans.transform.translation.y
        
        print(f"Robot is at: ({robot_x:.2f}, {robot_y:.2f})")
        
        # Send goal 1 meter ahead
        goal_pub = rospy.Publisher('/robot1/move_base_simple/goal', PoseStamped, queue_size=1)
        rospy.sleep(1.0)
        
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = 'robot1_map'
        goal.pose.position.x = robot_x + 1.0
        goal.pose.position.y = robot_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        
        print(f"Sending goal to: ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})")
        
        goal_pub.publish(goal)
        print("Goal sent!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    try:
        send_goal()
        rospy.sleep(2.0)
    except rospy.ROSInterruptException:
        pass