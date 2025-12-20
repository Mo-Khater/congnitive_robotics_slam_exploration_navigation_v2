#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

class SimpleGoalSender:
    """Just sends robot to goal location after delay"""
    def __init__(self):
        rospy.init_node('simple_goal_sender')
        
        # Goal location
        self.goal_x = rospy.get_param('~goal_x', 5.0)
        self.goal_y = rospy.get_param('~goal_y', 5.0)
        
        # Delay before sending goal
        self.delay = rospy.get_param('~delay', 60.0)
        
        # Publisher
        self.goal_pub = rospy.Publisher('/robot1/goal', PoseStamped, queue_size=10)
        
        # Send goal after delay
        rospy.Timer(rospy.Duration(self.delay), self.send_goal, oneshot=True)
        
        rospy.loginfo(f"Will send goal to ({self.goal_x}, {self.goal_y}) after {self.delay}s")
    
    def send_goal(self, event):
        """Send navigation goal"""
        goal = PoseStamped()
        goal.header.frame_id = "robot1_map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = self.goal_x
        goal.pose.position.y = self.goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
        rospy.loginfo("="*70)
        rospy.loginfo(f"🎯 Goal sent to ({self.goal_x:.2f}, {self.goal_y:.2f})")
        rospy.loginfo("="*70)

if __name__ == '__main__':
    try:
        sender = SimpleGoalSender()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass