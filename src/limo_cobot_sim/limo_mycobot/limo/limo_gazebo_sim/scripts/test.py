#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import tf2_ros

class GoalManager:
    """Manage navigation goals and monitor goal achievement"""
    def __init__(self):
        rospy.init_node('goal_manager')
        
        # Parameters
        self.goal_tolerance_xy = rospy.get_param('~goal_tolerance_xy', 0.2)
        self.goal_tolerance_yaw = rospy.get_param('~goal_tolerance_yaw', 0.2)
        self.max_goal_time = rospy.get_param('~max_goal_time', 60.0)
        
        self.current_goal = None
        self.goal_start_time = None
        self.path = None
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        self.goal_sub = rospy.Subscriber('/robot1/goal', PoseStamped, self.goal_callback)
        self.path_sub = rospy.Subscriber('/robot1/global_path', Path, self.path_callback)
        
        # Publishers
        self.goal_reached_pub = rospy.Publisher('/robot1/goal_reached', Bool, queue_size=10)
        self.active_goal_pub = rospy.Publisher('/robot1/active_goal', PoseStamped, queue_size=10)
        
        # Timer for monitoring
        self.monitor_timer = rospy.Timer(rospy.Duration(0.2), self.monitor_goal)
        
        rospy.loginfo("Goal Manager initialized")
        
    def goal_callback(self, msg):
        """Receive new goal"""
        self.current_goal = msg
        self.goal_start_time = rospy.Time.now()
        rospy.loginfo(f"New goal received at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        
    def path_callback(self, msg):
        """Receive path"""
        self.path = msg
        
    def get_robot_pose(self):
        """Get current robot pose"""
        try:
            trans = self.tf_buffer.lookup_transform('robot1_map', 'robot1_base_link',
                                                   rospy.Time(0), rospy.Duration(0.1))
            
            pose = PoseStamped()
            pose.header.frame_id = 'robot1_map'
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.orientation = trans.transform.rotation
            
            return pose
        except:
            return None
            
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        return np.sqrt(dx**2 + dy**2)
        
    def calculate_yaw_difference(self, pose1, pose2):
        """Calculate yaw difference between two poses"""
        # Extract yaw from quaternions
        def quaternion_to_yaw(q):
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            return np.arctan2(siny_cosp, cosy_cosp)
            
        yaw1 = quaternion_to_yaw(pose1.pose.orientation)
        yaw2 = quaternion_to_yaw(pose2.pose.orientation)
        
        diff = abs(yaw1 - yaw2)
        if diff > np.pi:
            diff = 2 * np.pi - diff
            
        return diff
        
    def is_goal_reached(self, robot_pose):
        """Check if goal is reached"""
        if self.current_goal is None or robot_pose is None:
            return False
            
        # Check position tolerance
        distance = self.calculate_distance(robot_pose, self.current_goal)
        
        if distance > self.goal_tolerance_xy:
            return False
            
        # Check orientation tolerance (optional)
        # yaw_diff = self.calculate_yaw_difference(robot_pose, self.current_goal)
        # if yaw_diff > self.goal_tolerance_yaw:
        #     return False
            
        return True
        
    def is_goal_timeout(self):
        """Check if goal has timed out"""
        if self.goal_start_time is None:
            return False
            
        elapsed = (rospy.Time.now() - self.goal_start_time).to_sec()
        return elapsed > self.max_goal_time
        
    def monitor_goal(self, event):
        """Monitor goal achievement"""
        if self.current_goal is None:
            return
            
        # Publish active goal for visualization
        self.active_goal_pub.publish(self.current_goal)
        
        # Get robot pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
            
        # Check if goal reached
        if self.is_goal_reached(robot_pose):
            rospy.loginfo("Goal reached!")
            self.goal_reached_pub.publish(Bool(data=True))
            self.current_goal = None
            self.goal_start_time = None
            return
            
        # Check timeout
        if self.is_goal_timeout():
            rospy.logwarn("Goal timeout! Canceling goal.")
            self.current_goal = None
            self.goal_start_time = None
            self.goal_reached_pub.publish(Bool(data=False))
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        manager = GoalManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass


#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import tf2_ros
import tf2_geometry_msgs

class DWAPlanner:
    """Simplified DWA with goal abandonment and tolerance"""
    def __init__(self):
        rospy.init_node('local_planner')
        
        # Robot parameters
        self.max_vel_x = rospy.get_param('~max_vel_x', 0.5)
        self.min_vel_x = rospy.get_param('~min_vel_x', -0.2)
        self.max_vel_theta = rospy.get_param('~max_vel_theta', 1.0)
        self.max_acc_x = rospy.get_param('~max_acc_x', 0.5)
        self.max_acc_theta = rospy.get_param('~max_acc_theta', 1.0)
        self.v_resolution = rospy.get_param('~v_resolution', 0.1)
        self.w_resolution = rospy.get_param('~w_resolution', 0.2)
        self.dt = rospy.get_param('~dt', 0.2)
        self.predict_time = rospy.get_param('~predict_time', 1.5)
        self.robot_radius = rospy.get_param('~robot_radius', 0.15)
        
        # Path following parameters
        self.lookahead_min = 0.3
        self.lookahead_max = 1.0
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.4)  # More lenient!
        
        # Cost weights
        self.goal_cost_weight = 1.0
        self.speed_cost_weight = 0.5
        self.obstacle_cost_weight = 2.0
        
        self.current_vel = [0.0, 0.0]
        self.global_path = None
        self.scan_data = None
        self.current_pose = None
        self.path_received_time = None
        
        # Goal tracking for abandonment
        self.closest_distance_to_goal = float('inf')
        self.time_at_goal = None
        self.time_pursuing_goal = None
        self.max_pursuit_time = 30.0  # Give up after 30 seconds
        
        # Track if moving away
        self.distance_history = []
        self.position_history = []
        self.stuck_position_threshold = 0.3  # If robot moves < 30cm in 5 seconds, it's stuck
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        self.path_sub = rospy.Subscriber('/robot1/global_path', Path, self.path_callback)
        self.odom_sub = rospy.Subscriber('/robot1/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/robot1/scan', LaserScan, self.scan_callback)
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
        self.local_path_pub = rospy.Publisher('/robot1/local_path', Path, queue_size=10)
        self.goal_reached_pub = rospy.Publisher('/robot1/goal_reached', Bool, queue_size=10)
        self.goal_failed_pub = rospy.Publisher('/robot1/goal_failed', Bool, queue_size=10)
        
        # Control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Smart Local Planner (DWA with goal abandonment) initialized")
        
    def path_callback(self, msg):
        self.global_path = msg
        self.path_received_time = rospy.Time.now()
        self.time_pursuing_goal = rospy.Time.now()
        self.closest_distance_to_goal = float('inf')
        self.distance_history = []
        self.position_history = []
        self.time_at_goal = None
        rospy.loginfo(f"📍 New path received with {len(msg.poses)} waypoints")
        
    def odom_callback(self, msg):
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self.current_pose = msg.pose.pose
        
    def scan_callback(self, msg):
        self.scan_data = msg
        
    def get_dynamic_window(self):
        """Calculate dynamic window"""
        v, w = self.current_vel
        
        v_min = max(self.min_vel_x, v - self.max_acc_x * self.dt)
        v_max = min(self.max_vel_x, v + self.max_acc_x * self.dt)
        w_min = max(-self.max_vel_theta, w - self.max_acc_theta * self.dt)
        w_max = min(self.max_vel_theta, w + self.max_acc_theta * self.dt)
            
        return [v_min, v_max, w_min, w_max]
        
    def predict_trajectory(self, v, w):
        """Predict trajectory for given velocities"""
        trajectory = []
        x, y, theta = 0, 0, 0
        
        time = 0
        while time <= self.predict_time:
            trajectory.append([x, y, theta])
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            time += self.dt
            
        return np.array(trajectory)
    
    def calc_obstacle_cost(self, trajectory):
        """Calculate cost based on obstacles"""
        if self.scan_data is None:
            return 0.0
        
        obstacles = []
        for i, r in enumerate(self.scan_data.ranges):
            if r < self.scan_data.range_min or r > self.scan_data.range_max:
                continue
            if np.isnan(r) or np.isinf(r):
                continue
            
            angle = self.scan_data.angle_min + i * self.scan_data.angle_increment
            obs_x = r * np.cos(angle)
            obs_y = r * np.sin(angle)
            obstacles.append((obs_x, obs_y))
        
        if len(obstacles) == 0:
            return 0.0
        
        obstacles = np.array(obstacles)
        min_dist = float('inf')
        
        for point in trajectory:
            x, y = point[0], point[1]
            dists = np.sqrt((obstacles[:, 0] - x)**2 + (obstacles[:, 1] - y)**2)
            min_dist_at_point = np.min(dists)
            min_dist = min(min_dist, min_dist_at_point)
        
        safety_margin = self.robot_radius * 1.2
        if min_dist < safety_margin:
            return float('inf')
        
        danger_zone = 0.6
        if min_dist < danger_zone:
            cost = (danger_zone - min_dist) / (danger_zone - safety_margin)
            return cost * 0.5
        
        return 0.0
    
    def get_adaptive_lookahead_point(self):
        """Get lookahead point with adaptive distance"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        
        current_speed = abs(self.current_vel[0])
        lookahead_dist = self.lookahead_min + (self.lookahead_max - self.lookahead_min) * (current_speed / self.max_vel_x)
        
        try:
            best_point = None
            best_dist_diff = float('inf')
            
            for pose in self.global_path.poses:
                p_current = PoseStamped()
                p_current.header.frame_id = pose.header.frame_id
                p_current.header.stamp = rospy.Time(0)
                p_current.pose = pose.pose
                
                try:
                    p_base = self.tf_buffer.transform(p_current, 'robot1_base_link', 
                                                     rospy.Duration(0.1))
                except:
                    continue
                
                x = p_base.pose.position.x
                y = p_base.pose.position.y
                dist = np.sqrt(x**2 + y**2)
                
                if x > 0:
                    dist_diff = abs(dist - lookahead_dist)
                    if dist_diff < best_dist_diff:
                        best_dist_diff = dist_diff
                        best_point = p_base
            
            if best_point is None and len(self.global_path.poses) > 0:
                goal = self.global_path.poses[-1]
                goal_current = PoseStamped()
                goal_current.header.frame_id = goal.header.frame_id
                goal_current.header.stamp = rospy.Time(0)
                goal_current.pose = goal.pose
                
                try:
                    best_point = self.tf_buffer.transform(goal_current, 'robot1_base_link',
                                                         rospy.Duration(0.1))
                except:
                    pass
            
            return best_point
            
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Lookahead error: {e}")
            return None
        
    def calc_goal_cost(self, trajectory):
        """Calculate cost based on heading toward target"""
        target = self.get_adaptive_lookahead_point()
        
        if target is None:
            return 2.0
            
        target_x = target.pose.position.x
        target_y = target.pose.position.y
        
        target_angle = np.arctan2(target_y, target_x)
        final_angle = trajectory[-1, 2]
        angle_diff = abs(self.normalize_angle(target_angle - final_angle))
        
        final_x = trajectory[-1, 0]
        final_y = trajectory[-1, 1]
        final_dist = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        
        cost = angle_diff * 0.6 + final_dist * 0.4
        
        return cost
        
    def calc_speed_cost(self, v):
        """Prefer forward motion at reasonable speed"""
        if v < 0:
            return 1.5
        
        target_speed = self.max_vel_x * 0.7
        speed_diff = abs(v - target_speed) / self.max_vel_x
        
        return speed_diff
        
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def calc_trajectory_cost(self, v, w):
        """Calculate total cost"""
        trajectory = self.predict_trajectory(v, w)
        
        obstacle_cost = self.calc_obstacle_cost(trajectory)
        if obstacle_cost == float('inf'):
            return float('inf')
        
        goal_cost = self.calc_goal_cost(trajectory)
        speed_cost = self.calc_speed_cost(v)
        
        total_cost = (
            self.obstacle_cost_weight * obstacle_cost +
            self.goal_cost_weight * goal_cost +
            self.speed_cost_weight * speed_cost
        )
        
        return total_cost
        
    def dwa_control(self):
        """DWA control"""
        dw = self.get_dynamic_window()
        
        best_v, best_w = 0.0, 0.0
        min_cost = float('inf')
        best_trajectory = None
        
        v_samples = np.arange(dw[0], dw[1], self.v_resolution)
        w_samples = np.arange(dw[2], dw[3], self.w_resolution)
        
        if 0.0 not in v_samples and dw[0] <= 0.0 <= dw[1]:
            v_samples = np.append(v_samples, 0.0)
        if 0.0 not in w_samples and dw[2] <= 0.0 <= dw[3]:
            w_samples = np.append(w_samples, 0.0)
        
        valid_count = 0
        for v in v_samples:
            for w in w_samples:
                cost = self.calc_trajectory_cost(v, w)
                
                if cost != float('inf'):
                    valid_count += 1
                
                if cost < min_cost:
                    min_cost = cost
                    best_v = v
                    best_w = w
                    best_trajectory = self.predict_trajectory(v, w)
        
        if valid_count == 0:
            rospy.logwarn_throttle(2.0, "⚠️ No valid trajectories - rotation recovery")
            best_v = 0.0
            best_w = 0.6
            best_trajectory = self.predict_trajectory(best_v, best_w)
        
        return best_v, best_w, best_trajectory
        
    def distance_to_goal(self):
        """Calculate distance to final goal"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return float('inf')
            
        try:
            goal = self.global_path.poses[-1]
            goal_current = PoseStamped()
            goal_current.header.frame_id = goal.header.frame_id
            goal_current.header.stamp = rospy.Time(0)
            goal_current.pose = goal.pose
            
            goal_base = self.tf_buffer.transform(goal_current, 'robot1_base_link',
                                                rospy.Duration(0.1))
            dist = np.sqrt(goal_base.pose.position.x**2 + goal_base.pose.position.y**2)
            return dist
        except:
            return float('inf')
    
    def is_stuck(self):
        """Detect if robot is stuck (not moving)"""
        if len(self.position_history) < 25:  # Need 2.5 seconds of history
            return False
        
        recent_positions = self.position_history[-25:]
        
        # Calculate total distance moved in last 2.5 seconds
        total_movement = 0
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            total_movement += np.sqrt(dx**2 + dy**2)
        
        # If moved less than 30cm in 2.5 seconds, we're stuck
        if total_movement < self.stuck_position_threshold:
            return True
        
        return False
    
    def is_moving_away_from_goal(self, current_distance):
        """Detect if consistently moving away from goal"""
        self.distance_history.append(current_distance)
        if len(self.distance_history) > 20:
            self.distance_history.pop(0)
        
        if len(self.distance_history) < 15:
            return False
        
        # Check recent trend
        recent = self.distance_history[-15:]
        
        # Count how many times distance increased
        increasing = 0
        for i in range(1, len(recent)):
            if recent[i] > recent[i-1] + 0.03:  # Increasing by >3cm
                increasing += 1
        
        # If 10+ out of 15 show increase, we're moving away
        return increasing >= 10
        
    def control_loop(self, event):
        """Main control loop with smart goal abandonment"""
        # No path - stop
        if self.global_path is None or self.current_pose is None:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return
        
        # Track position history for stuck detection
        if self.current_pose is not None:
            pos = (self.current_pose.position.x, self.current_pose.position.y)
            self.position_history.append(pos)
            if len(self.position_history) > 50:  # Keep 5 seconds
                self.position_history.pop(0)
        
        # Get distance to goal
        dist_to_goal = self.distance_to_goal()
        
        # Update closest distance
        if dist_to_goal < self.closest_distance_to_goal:
            self.closest_distance_to_goal = dist_to_goal
        
        # CHECK 1: Are we close enough to goal? (TOLERANCE)
        if dist_to_goal < self.goal_tolerance:
            # Wait a bit to stabilize
            if self.time_at_goal is None:
                self.time_at_goal = rospy.Time.now()
            
            time_at_goal = (rospy.Time.now() - self.time_at_goal).to_sec()
            
            if time_at_goal > 0.5:  # Stable for 0.5 seconds
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                rospy.loginfo(f"✅ Goal reached! (dist={dist_to_goal:.2f}m, tolerance={self.goal_tolerance}m)")
                self.goal_reached_pub.publish(Bool(data=True))
                self.global_path = None
                return
        else:
            self.time_at_goal = None
        
        # CHECK 2: Timeout - been trying too long?
        if self.time_pursuing_goal is not None:
            pursuit_time = (rospy.Time.now() - self.time_pursuing_goal).to_sec()
            
            if pursuit_time > self.max_pursuit_time:
                rospy.logerr(f"❌ GOAL TIMEOUT! Been trying for {pursuit_time:.1f}s (max={self.max_pursuit_time}s)")
                rospy.logerr(f"   Current distance: {dist_to_goal:.2f}m, Closest: {self.closest_distance_to_goal:.2f}m")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.goal_failed_pub.publish(Bool(data=True))
                self.global_path = None
                return
        
        # CHECK 3: Are we stuck?
        if self.is_stuck():
            rospy.logerr(f"❌ ROBOT STUCK! Not moving for 2.5+ seconds")
            rospy.logerr(f"   Distance to goal: {dist_to_goal:.2f}m")
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.goal_failed_pub.publish(Bool(data=True))
            self.global_path = None
            return
        
        # CHECK 4: Moving away from goal?
        if dist_to_goal < 10.0:  # Only check if goal is reasonably close
            if self.is_moving_away_from_goal(dist_to_goal):
                rospy.logerr(f"❌ MOVING AWAY FROM GOAL!")
                rospy.logerr(f"   Current: {dist_to_goal:.2f}m, Closest: {self.closest_distance_to_goal:.2f}m")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.goal_failed_pub.publish(Bool(data=True))
                self.global_path = None
                return
        
        # CHECK 5: Got close but now far away? (overshot)
        if self.closest_distance_to_goal < self.goal_tolerance * 1.5:
            if dist_to_goal > self.closest_distance_to_goal + 0.5:  # Now 0.5m further
                rospy.logwarn(f"⚠️ OVERSHOT GOAL! Closest was {self.closest_distance_to_goal:.2f}m, now {dist_to_goal:.2f}m")
                rospy.logwarn(f"   Declaring goal reached at closest approach")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.goal_reached_pub.publish(Bool(data=True))
                self.global_path = None
                return
        
        # All checks passed - continue navigation
        v, w, trajectory = self.dwa_control()
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        
        # Debug info
        pursuit_time = (rospy.Time.now() - self.time_pursuing_goal).to_sec() if self.time_pursuing_goal else 0
        rospy.loginfo_throttle(1.0, 
            f"v={v:.2f} w={w:.2f} | dist={dist_to_goal:.2f}m | best={self.closest_distance_to_goal:.2f}m | t={pursuit_time:.1f}s")
        
        # Publish trajectory
        if trajectory is not None:
            self.publish_local_path(trajectory)
            
    def publish_local_path(self, trajectory):
        """Publish predicted trajectory"""
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "robot1_base_link"
        
        for point in trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.local_path_pub.publish(path_msg)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        planner = DWAPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass