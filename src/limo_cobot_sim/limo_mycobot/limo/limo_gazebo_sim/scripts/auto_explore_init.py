#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid

class AutoExplore:
    def __init__(self):
        rospy.init_node('auto_explore_init', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.map_received = False
        self.map_updates = 0
        
        # Subscribe to map to check if SLAM is working
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
    def map_callback(self, msg):
        self.map_received = True
        self.map_updates += 1
        
    def move_robot(self, linear_x, angular_z, duration):
        """Move robot with given velocities for duration seconds"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # Stop
        twist.linear.x = 0
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.2)
    
    def run(self):
        rospy.loginfo("Starting automatic exploration initialization...")
        rospy.sleep(2)  # Wait for publishers to connect
        
        # Drive forward
        rospy.loginfo("Driving forward...")
        self.move_robot(0.4, 0.0, 5)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Turn left
        rospy.loginfo("Turning left...")
        self.move_robot(0.0, 1.0, 3)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Drive forward
        rospy.loginfo("Driving forward...")
        self.move_robot(0.4, 0.0, 5)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Turn right
        rospy.loginfo("Turning right...")
        self.move_robot(0.0, -1.0, 3)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Drive forward
        rospy.loginfo("Driving forward...")
        self.move_robot(0.4, 0.0, 5)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Turn right again
        rospy.loginfo("Turning right...")
        self.move_robot(0.0, -1.0, 3)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Drive forward
        rospy.loginfo("Driving forward...")
        self.move_robot(0.4, 0.0, 5)
        rospy.loginfo(f"Map updates so far: {self.map_updates}")
        
        # Final turn
        rospy.loginfo("Final turn...")
        self.move_robot(0.0, 1.0, 3)
        
        rospy.loginfo(f"Initialization complete! Total map updates: {self.map_updates}")
        rospy.loginfo("Map should now be ready for autonomous exploration")
        
        if self.map_received:
            rospy.loginfo("✓ Map received successfully")
        else:
            rospy.logwarn("✗ WARNING: Map was not received. SLAM may not be working properly.")

if __name__ == '__main__':
    try:
        explorer = AutoExplore()
        explorer.run()
    except rospy.ROSInterruptException:
        pass