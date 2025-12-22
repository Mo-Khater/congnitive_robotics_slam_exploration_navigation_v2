#!/usr/bin/env python3
"""
Odometry to TF Broadcaster for LIMO Robot

CRITICAL NODE: The real LIMO robot publishes /odom topic but does NOT
publish the TF transform from odom to base_footprint. This node bridges
that gap by listening to /odom and broadcasting the corresponding TF.

Without this node, the navigation stack will fail because it cannot
determine the robot's position in the odometry frame.
"""

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped


class OdomToTF:
    def __init__(self):
        rospy.init_node('odom_to_tf', anonymous=False)
        
        # Parameters
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_footprint')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Subscribe to odometry
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, 
            Odometry, 
            self.odom_callback,
            queue_size=10
        )
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Odometry to TF Bridge Node Started")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Subscribing to: {self.odom_topic}")
        rospy.loginfo(f"Broadcasting TF: {self.odom_frame} -> {self.base_frame}")
        rospy.loginfo("=" * 60)
    
    def odom_callback(self, msg):
        """
        Callback for odometry messages.
        Converts odometry data to TF transform.
        """
        # Create transform message
        transform = TransformStamped()
        
        # Header
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        
        # Translation (position)
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        
        # Rotation (orientation)
        transform.transform.rotation.x = msg.pose.pose.orientation.x
        transform.transform.rotation.y = msg.pose.pose.orientation.y
        transform.transform.rotation.z = msg.pose.pose.orientation.z
        transform.transform.rotation.w = msg.pose.pose.orientation.w
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)
    
    def run(self):
        """Keep the node running"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = OdomToTF()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Odometry to TF node terminated.")