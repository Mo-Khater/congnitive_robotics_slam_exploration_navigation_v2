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
    """Improved DWA: Better path tracking with tighter curves"""
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
        
        # Path following - TIGHTER TRACKING
        self.lookahead_min = 0.2  # Reduced from 0.3
        self.lookahead_max = 0.5  # Reduced from 0.8
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.35)
        
        # Cost weights - PRIORITIZE PATH FOLLOWING
        self.goal_cost_weight = 3.0      # Much higher - follow path closely!
        self.speed_cost_weight = 0.3     # Lower priority
        self.obstacle_cost_weight = 1.5  # Moderate safety
        
        self.current_vel = [0.0, 0.0]
        self.global_path = None
        self.scan_data = None
        self.current_pose = None
        self.path_received_time = None
        
        # Smart goal management
        self.closest_distance_to_goal = float('inf')
        self.time_at_goal = None
        self.time_pursuing_goal = None
        self.max_pursuit_time = 40.0
        
        # Progress tracking
        self.distance_history = []
        self.position_history = []
        self.last_progress_check = rospy.Time.now()
        self.position_at_last_check = None
        
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
        
        rospy.loginfo("Improved Local Planner initialized - Better path tracking")
        
    def path_callback(self, msg):
        self.global_path = msg
        self.path_received_time = rospy.Time.now()
        self.time_pursuing_goal = rospy.Time.now()
        self.closest_distance_to_goal = float('inf')
        self.distance_history = []
        self.position_history = []
        self.time_at_goal = None
        self.last_progress_check = rospy.Time.now()
        self.position_at_last_check = None
        rospy.loginfo(f"New path: {len(msg.poses)} waypoints")
        
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
        """Predict trajectory"""
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
        """Obstacle cost - balanced"""
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
        
        # Safety margin
        safety_margin = self.robot_radius * 0.95
        if min_dist < safety_margin:
            return float('inf')
        
        # Soft cost
        danger_zone = 0.4
        if min_dist < danger_zone:
            cost = (danger_zone - min_dist) / (danger_zone - safety_margin)
            return cost * 0.5
        
        return 0.0
    
    def get_closest_path_point(self):
        """Get the closest point on the path ahead of robot"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        
        try:
            closest_point = None
            min_dist = float('inf')
            
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
                
                # Only consider points ahead
                if x > 0:
                    dist = np.sqrt(x**2 + y**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = p_base
            
            return closest_point
            
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Closest point error: {e}")
            return None
    
    def get_lookahead_point(self):
        """Get lookahead point - SHORT distance for tight tracking"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        
        # Use speed-dependent lookahead but keep it short
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
                
                # Find point at lookahead distance
                if x > 0:
                    dist_diff = abs(dist - lookahead_dist)
                    if dist_diff < best_dist_diff:
                        best_dist_diff = dist_diff
                        best_point = p_base
            
            if best_point is not None:
                return best_point
            
            # Fallback: use closest point
            return self.get_closest_path_point()
            
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Lookahead error: {e}")
            return None
        
    def calc_goal_cost(self, trajectory):
        """Goal cost - STRONG emphasis on following path direction"""
        target = self.get_lookahead_point()
        
        if target is None:
            return 5.0
            
        target_x = target.pose.position.x
        target_y = target.pose.position.y
        
        # Angle to target
        target_angle = np.arctan2(target_y, target_x)
        final_angle = trajectory[-1, 2]
        angle_diff = abs(self.normalize_angle(target_angle - final_angle))
        
        # Distance to target
        final_x = trajectory[-1, 0]
        final_y = trajectory[-1, 1]
        final_dist = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        
        # HEAVILY weight angle error to follow path direction
        cost = angle_diff * 1.5 + final_dist * 0.3
        
        # Additional penalty for large lateral deviation
        # Cross-track error: perpendicular distance from trajectory endpoint to target line
        if target_x > 0.01:  # Avoid division by zero
            # Calculate perpendicular distance from trajectory end to target direction
            cross_track = abs(final_y * np.cos(target_angle) - final_x * np.sin(target_angle))
            cost += cross_track * 2.0  # Heavy penalty for going sideways
        
        return cost
        
    def calc_speed_cost(self, v):
        """Speed cost - prefer forward movement"""
        if v < 0:
            return 2.0
        
        # Prefer moderate speed (60%)
        target_speed = self.max_vel_x * 0.6
        speed_diff = abs(v - target_speed) / self.max_vel_x
        
        return speed_diff * 0.8
        
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def calc_trajectory_cost(self, v, w):
        """Total cost"""
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
        
        # Ensure zero velocity is an option
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
        
        # Recovery if blocked
        if valid_count == 0:
            rospy.logwarn_throttle(2.0, "No valid trajectories - recovery rotation")
            best_v = 0.0
            best_w = 0.6
            best_trajectory = self.predict_trajectory(best_v, best_w)
        elif valid_count < 3:
            # Very limited options - ensure some movement
            if abs(best_v) < 0.05:
                rospy.loginfo_throttle(2.0, "Limited options, forcing minimum velocity")
                best_v = max(best_v, 0.15)
        
        return best_v, best_w, best_trajectory
        
    def distance_to_goal(self):
        """Distance to goal"""
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
    
    def check_making_progress(self):
        """Check if robot is making progress toward goal"""
        if self.current_pose is None:
            return True
        
        now = rospy.Time.now()
        time_since_check = (now - self.last_progress_check).to_sec()
        
        # Check every 5 seconds
        if time_since_check < 5.0:
            return True
        
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])
        
        if self.position_at_last_check is None:
            self.position_at_last_check = current_pos
            self.last_progress_check = now
            return True
        
        # Calculate movement in 5 seconds
        movement = np.linalg.norm(current_pos - self.position_at_last_check)
        
        self.position_at_last_check = current_pos
        self.last_progress_check = now
        
        # Need at least 40cm in 5 seconds
        if movement < 0.4:
            rospy.logwarn(f"Poor progress: moved only {movement:.2f}m in 5 seconds")
            return False
        
        return True
        
    def control_loop(self, event):
        """Main control"""
        # No path - stop
        if self.global_path is None or self.current_pose is None:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return
        
        # Get distance to goal
        dist_to_goal = self.distance_to_goal()
        
        # Update closest
        if dist_to_goal < self.closest_distance_to_goal:
            self.closest_distance_to_goal = dist_to_goal
        
        # GOAL REACHED
        if dist_to_goal < self.goal_tolerance:
            if self.time_at_goal is None:
                self.time_at_goal = rospy.Time.now()
            
            time_at_goal = (rospy.Time.now() - self.time_at_goal).to_sec()
            
            if time_at_goal > 0.5:
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                rospy.loginfo(f"✅ Goal reached (dist={dist_to_goal:.2f}m)")
                
                for _ in range(3):
                    self.goal_reached_pub.publish(Bool(data=True))
                    rospy.sleep(0.1)
                
                self.global_path = None
                return
        else:
            self.time_at_goal = None
        
        # TIMEOUT CHECK
        if self.time_pursuing_goal is not None:
            pursuit_time = (rospy.Time.now() - self.time_pursuing_goal).to_sec()
            
            if pursuit_time > self.max_pursuit_time:
                rospy.logerr(f"❌ Timeout after {pursuit_time:.1f}s")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                
                for _ in range(3):
                    self.goal_failed_pub.publish(Bool(data=True))
                    rospy.sleep(0.1)
                
                self.global_path = None
                return
        
        # PROGRESS CHECK
        if not self.check_making_progress():
            if not hasattr(self, 'no_progress_count'):
                self.no_progress_count = 0
            self.no_progress_count += 1
            
            # Give 3 chances (15 seconds total)
            if self.no_progress_count >= 3:
                rospy.logerr("❌ No progress - stuck or unreachable")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                
                for _ in range(3):
                    self.goal_failed_pub.publish(Bool(data=True))
                    rospy.sleep(0.1)
                
                self.global_path = None
                return
        else:
            self.no_progress_count = 0
        
        # OVERSHOT CHECK
        if self.closest_distance_to_goal < self.goal_tolerance * 1.2:
            if dist_to_goal > self.closest_distance_to_goal + 0.6:
                rospy.logwarn(f"Overshot goal")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                
                for _ in range(3):
                    self.goal_reached_pub.publish(Bool(data=True))
                    rospy.sleep(0.1)
                
                self.global_path = None
                return
        
        # DWA CONTROL
        v, w, trajectory = self.dwa_control()
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        
        # Debug
        pursuit_time = (rospy.Time.now() - self.time_pursuing_goal).to_sec() if self.time_pursuing_goal else 0
        rospy.loginfo_throttle(1.0, 
            f"robot1 v={v:.2f} w={w:.2f} | dist={dist_to_goal:.2f}m | t={pursuit_time:.0f}s")
        
        # Visualize
        if trajectory is not None:
            self.publish_local_path(trajectory)
            
    def publish_local_path(self, trajectory):
        """Publish trajectory visualization"""
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