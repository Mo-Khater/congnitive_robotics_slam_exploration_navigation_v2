#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import tf2_ros

class GoalManager:
    """Simplified goal manager - local planner handles most logic"""
    def __init__(self):
        rospy.init_node('goal_manager')
        
        # Parameters
        self.goal_tolerance_xy = rospy.get_param('~goal_tolerance_xy', 0.35)
        self.max_goal_time = rospy.get_param('~max_goal_time', 45.0)
        
        self.current_goal = None
        self.goal_start_time = None
        self.path = None
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        self.goal_sub = rospy.Subscriber('/goal', PoseStamped, self.goal_callback)
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback)
        
        # Publishers
        self.goal_reached_pub = rospy.Publisher('/goal_reached', Bool, queue_size=10)
        self.active_goal_pub = rospy.Publisher('/active_goal', PoseStamped, queue_size=10)
        
        # Timer for monitoring
        self.monitor_timer = rospy.Timer(rospy.Duration(0.2), self.monitor_goal)
        
        rospy.loginfo("Goal Manager initialized (simple mode)")
        
    def goal_callback(self, msg):
        """Receive new goal"""
        self.current_goal = msg
        self.goal_start_time = rospy.Time.now()
        rospy.loginfo(f"Goal received at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        
    def path_callback(self, msg):
        """Receive path"""
        self.path = msg
        
    def get_robot_pose(self):
        """Get current robot pose"""
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link',
                                                   rospy.Time(0), rospy.Duration(0.1))
            
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.orientation = trans.transform.rotation
            
            return pose
        except:
            return None
            
    def calculate_distance(self, pose1, pose2):
        """Calculate distance between poses"""
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        return np.sqrt(dx**2 + dy**2)
        
    def is_goal_reached(self, robot_pose):
        """Check if goal reached"""
        if self.current_goal is None or robot_pose is None:
            return False
            
        distance = self.calculate_distance(robot_pose, self.current_goal)
        return distance < self.goal_tolerance_xy
        
    def is_goal_timeout(self):
        """Check timeout"""
        if self.goal_start_time is None:
            return False
            
        elapsed = (rospy.Time.now() - self.goal_start_time).to_sec()
        return elapsed > self.max_goal_time
        
    def monitor_goal(self, event):
        """Monitor goal - simple backup check"""
        if self.current_goal is None:
            return
            
        # Publish for visualization
        self.active_goal_pub.publish(self.current_goal)
        
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
            
        # Simple check - local planner does most work
        if self.is_goal_reached(robot_pose):
            rospy.loginfo("Goal reached (backup check)")
            self.goal_reached_pub.publish(Bool(data=True))
            self.current_goal = None
            self.goal_start_time = None
            return
            
        # Timeout backup
        if self.is_goal_timeout():
            rospy.logwarn("Goal timeout (backup check)")
            self.current_goal = None
            self.goal_start_time = None
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        manager = GoalManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass