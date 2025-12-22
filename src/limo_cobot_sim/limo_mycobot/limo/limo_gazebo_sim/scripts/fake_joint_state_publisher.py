#!/usr/bin/env python3
"""
Fake Joint State Publisher for LIMO Robot

The real LIMO robot doesn't publish /joint_states, which are needed
for RViz to visualize the wheels correctly. This node publishes
fake joint states for the four wheels.
"""

import rospy
from sensor_msgs.msg import JointState


class FakeJointStatePublisher:
    def __init__(self):
        rospy.init_node('fake_joint_state_publisher', anonymous=False)
        
        # Publisher
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        
        # Joint names for LIMO four-wheel differential drive
        self.joint_names = [
            'front_left_wheel',
            'front_right_wheel',
            'rear_left_wheel',
            'rear_right_wheel'
        ]
        
        # Publishing rate
        self.rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Fake Joint State Publisher initialized")
        rospy.loginfo(f"Publishing joint states for: {self.joint_names}")
    
    def publish_joint_states(self):
        """Publish fake joint states for visualization"""
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.joint_names
        
        # Set all positions to 0.0 (we don't track actual wheel rotations)
        joint_state.position = [0.0] * len(self.joint_names)
        
        # Optional: Set velocities and effort to empty lists
        joint_state.velocity = []
        joint_state.effort = []
        
        self.joint_pub.publish(joint_state)
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Starting to publish fake joint states...")
        
        while not rospy.is_shutdown():
            self.publish_joint_states()
            self.rate.sleep()


if __name__ == '__main__':
    try:
        publisher = FakeJointStatePublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Fake Joint State Publisher node terminated.")