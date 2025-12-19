#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import tf2_ros

class PoseTracker:
    """Simple pose tracker that publishes robot pose from TF"""
    def __init__(self):
        rospy.init_node('pose_tracker')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.pose_pub = rospy.Publisher('/robot3/robot_pose', PoseStamped, queue_size=10)
        
        # Timer for publishing pose
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_pose)
        
        rospy.loginfo("Pose Tracker initialized")
        
    def publish_pose(self, event):
        """Publish current robot pose"""
        try:
            trans = self.tf_buffer.lookup_transform('robot3_map', 'robot3_base_link',
                                                   rospy.Time(0), rospy.Duration(0.1))
            
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'robot3_map'
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            
            self.pose_pub.publish(pose)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            pass
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker = PoseTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass