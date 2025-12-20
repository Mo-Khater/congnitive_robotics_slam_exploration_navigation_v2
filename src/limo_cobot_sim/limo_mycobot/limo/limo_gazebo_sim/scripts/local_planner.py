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
    """Improved DWA: Faster, smarter door navigation"""
    def __init__(self):
        rospy.init_node('local_planner')
        
        # Robot parameters - Balanced: Fast but accurate
        self.max_vel_x = rospy.get_param('~max_vel_x', 0.65)  # Moderate speed
        self.min_vel_x = rospy.get_param('~min_vel_x', -0.25)
        self.max_vel_theta = rospy.get_param('~max_vel_theta', 1.2)  # Moderate rotation
        self.max_acc_x = rospy.get_param('~max_acc_x', 0.6)  # Smooth acceleration
        self.max_acc_theta = rospy.get_param('~max_acc_theta', 1.2)
        self.v_resolution = rospy.get_param('~v_resolution', 0.1)  # Finer resolution
        self.w_resolution = rospy.get_param('~w_resolution', 0.15)  # Finer resolution
        self.dt = rospy.get_param('~dt', 0.2)
        self.predict_time = rospy.get_param('~predict_time', 1.5)
        self.robot_radius = rospy.get_param('~robot_radius', 0.15)
        
        # Path following - Adaptive lookahead
        self.lookahead_min = 0.4   # Closer tracking
        self.lookahead_max = 1.2   # Still allows smooth curves
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.3)  # Tighter tolerance
        
        # Cost weights - Conservative balance for accuracy
        self.goal_cost_weight = 1.5      # Higher - follow path closely
        self.speed_cost_weight = 0.6     # Moderate - still encourage movement
        self.obstacle_cost_weight = 1.3  # Higher - be more careful
        
        self.current_vel = [0.0, 0.0]
        self.global_path = None
        self.scan_data = None
        self.current_pose = None
        self.path_received_time = None
        
        # Smart goal management
        self.closest_distance_to_goal = float('inf')
        self.time_at_goal = None
        self.time_pursuing_goal = None
        self.max_pursuit_time = 50.0  # More time for difficult navigation
        
        # Progress tracking
        self.distance_history = []
        self.position_history = []
        self.last_progress_check = rospy.Time.now()
        self.position_at_last_check = None
        
        # Stuck detection - More patient
        self.consecutive_low_velocity = 0
        self.stuck_threshold = 20  # 2.0 seconds (20 * 0.1s) - more patient
        self.rotation_recovery_active = False
        self.recovery_start_time = None
        self.max_recovery_time = 2.5  # Shorter recovery attempts
        
        # Oscillation detection
        self.angular_velocity_history = []
        self.max_angular_history = 20
        
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
        
        rospy.loginfo("Conservative DWA Planner - Balanced speed and accuracy for precise navigation")
        
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
        self.consecutive_low_velocity = 0
        self.rotation_recovery_active = False
        self.angular_velocity_history = []
        rospy.loginfo(f"New path: {len(msg.poses)} waypoints")
        
    def odom_callback(self, msg):
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self.current_pose = msg.pose.pose
        
        # Track angular velocity for oscillation detection
        self.angular_velocity_history.append(msg.twist.twist.angular.z)
        if len(self.angular_velocity_history) > self.max_angular_history:
            self.angular_velocity_history.pop(0)
        
    def scan_callback(self, msg):
        self.scan_data = msg
        
    def detect_oscillation(self):
        """Detect if robot is oscillating left-right"""
        if len(self.angular_velocity_history) < 10:
            return False
        
        recent_angular = self.angular_velocity_history[-10:]
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(recent_angular)):
            if abs(recent_angular[i]) > 0.2 and abs(recent_angular[i-1]) > 0.2:
                if np.sign(recent_angular[i]) != np.sign(recent_angular[i-1]):
                    sign_changes += 1
        
        # If changing direction more than 3 times in 10 samples, we're oscillating
        return sign_changes >= 3
        
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
        """Obstacle cost - Conservative safety margins"""
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
        
        # Conservative safety margin
        safety_margin = self.robot_radius * 1.4
        if min_dist < safety_margin:
            return float('inf')
        
        # Gradual cost increase with larger danger zone
        danger_zone = 0.7
        if min_dist < danger_zone:
            cost = (danger_zone - min_dist) / (danger_zone - safety_margin)
            return cost * 0.6  # Moderate penalty
        
        return 0.0
    
    def get_lookahead_point(self):
        """Get lookahead point - adaptive based on speed"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        
        # Speed-dependent lookahead - faster = look further ahead
        current_speed = abs(self.current_vel[0])
        speed_ratio = current_speed / self.max_vel_x
        lookahead_dist = self.lookahead_min + (self.lookahead_max - self.lookahead_min) * speed_ratio
        
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
                
                # Find point closest to lookahead distance, ahead of robot
                if x > 0:
                    dist_diff = abs(dist - lookahead_dist)
                    if dist_diff < best_dist_diff:
                        best_dist_diff = dist_diff
                        best_point = p_base
            
            return best_point
            
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Lookahead error: {e}")
            return None
        
    def calc_goal_cost(self, trajectory):
        """Goal cost - Precise path following"""
        target = self.get_lookahead_point()
        
        if target is None:
            return 5.0
            
        target_x = target.pose.position.x
        target_y = target.pose.position.y
        target_dist = np.sqrt(target_x**2 + target_y**2)
        
        # Angle to target
        target_angle = np.arctan2(target_y, target_x)
        final_angle = trajectory[-1, 2]
        angle_diff = abs(self.normalize_angle(target_angle - final_angle))
        
        # Distance to target after trajectory
        final_x = trajectory[-1, 0]
        final_y = trajectory[-1, 1]
        final_dist = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        
        # Prioritize both heading and reaching the target
        cost = angle_diff * 1.0 + final_dist * 1.0
        
        # Moderate penalty for lateral deviation - keep on path
        if target_x > 0.01:
            cross_track = abs(final_y * np.cos(target_angle) - final_x * np.sin(target_angle))
            cost += cross_track * 0.6  # Encourage path following
        
        return cost
        
    def calc_speed_cost(self, v):
        """Speed cost - Moderate speed preference"""
        if v < 0:
            return 3.0  # Strong penalty for backwards
        
        # Prefer moderate speed (60% of max) for better control
        target_speed = self.max_vel_x * 0.6
        speed_diff = abs(v - target_speed) / self.max_vel_x
        
        return speed_diff * 0.6
        
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
        """DWA control with recovery logic"""
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
            rospy.logwarn_throttle(1.0, "⚠️ BLOCKED - No valid trajectories!")
            best_v = 0.0
            best_w = 1.0 if not hasattr(self, 'recovery_direction') else self.recovery_direction
            self.recovery_direction = best_w
            best_trajectory = self.predict_trajectory(best_v, best_w)
        
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
        """Check if robot is making progress"""
        if self.current_pose is None:
            return True
        
        now = rospy.Time.now()
        time_since_check = (now - self.last_progress_check).to_sec()
        
        # Check every 4 seconds
        if time_since_check < 4.0:
            return True
        
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])
        
        if self.position_at_last_check is None:
            self.position_at_last_check = current_pos
            self.last_progress_check = now
            return True
        
        # Calculate movement
        movement = np.linalg.norm(current_pos - self.position_at_last_check)
        
        self.position_at_last_check = current_pos
        self.last_progress_check = now
        
        # Need at least 30cm in 4 seconds
        min_required = 0.30
        if abs(self.current_vel[1]) > 0.4:  # If rotating significantly
            min_required = 0.20
        
        if movement < min_required:
            rospy.logwarn(f"⚠️ Poor progress: moved only {movement:.2f}m in 4 seconds")
            return False
        
        return True
        
    def control_loop(self, event):
        """Main control loop"""
        # No path - stop
        if self.global_path is None or self.current_pose is None:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.rotation_recovery_active = False
            return
        
        # Get distance to goal
        dist_to_goal = self.distance_to_goal()
        
        # Update closest distance
        if dist_to_goal < self.closest_distance_to_goal:
            self.closest_distance_to_goal = dist_to_goal
        
        # GOAL REACHED - More precise check
        if dist_to_goal < self.goal_tolerance:
            if self.time_at_goal is None:
                self.time_at_goal = rospy.Time.now()
            
            time_at_goal = (rospy.Time.now() - self.time_at_goal).to_sec()
            
            # Need to be stable at goal for 0.8 seconds
            if time_at_goal > 0.8:
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                rospy.loginfo(f"✅ Goal reached (dist={dist_to_goal:.2f}m)")
                
                for _ in range(3):
                    self.goal_reached_pub.publish(Bool(data=True))
                    rospy.sleep(0.1)
                
                self.global_path = None
                self.rotation_recovery_active = False
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
                self.rotation_recovery_active = False
                return
        
        # OSCILLATION DETECTION
        if self.detect_oscillation():
            rospy.logwarn_throttle(2.0, "🔄 Oscillation detected - trying to break the pattern")
            # Force a decisive move: stop rotation, push forward if possible
            if not self.rotation_recovery_active:
                self.rotation_recovery_active = True
                self.recovery_start_time = rospy.Time.now()
                self.angular_velocity_history = []  # Clear history
        
        # Check if recovery is taking too long
        if self.rotation_recovery_active and self.recovery_start_time is not None:
            recovery_duration = (rospy.Time.now() - self.recovery_start_time).to_sec()
            if recovery_duration > self.max_recovery_time:
                rospy.loginfo("Recovery rotation complete")
                self.rotation_recovery_active = False
                self.recovery_start_time = None
        
        # PROGRESS CHECK
        if not self.check_making_progress():
            if not hasattr(self, 'no_progress_count'):
                self.no_progress_count = 0
            self.no_progress_count += 1
            
            # Give 3 chances (12 seconds total)
            if self.no_progress_count >= 3:
                rospy.logerr("❌ No progress - goal unreachable")
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                
                for _ in range(3):
                    self.goal_failed_pub.publish(Bool(data=True))
                    rospy.sleep(0.1)
                
                self.global_path = None
                self.no_progress_count = 0
                self.rotation_recovery_active = False
                return
        else:
            self.no_progress_count = 0
        
        # DWA CONTROL
        v, w, trajectory = self.dwa_control()
        
        # Stuck detection with smarter recovery
        is_stuck = abs(v) < 0.08 and abs(w) < 0.12  # Tighter thresholds
        
        if is_stuck:
            self.consecutive_low_velocity += 1
            
            if self.consecutive_low_velocity >= self.stuck_threshold:
                if not self.rotation_recovery_active:
                    rospy.logwarn(f"🚨 STUCK! Initiating recovery rotation")
                    self.rotation_recovery_active = True
                    self.recovery_start_time = rospy.Time.now()
                    if not hasattr(self, 'recovery_direction'):
                        self.recovery_direction = 1.0
                    else:
                        # Alternate direction
                        self.recovery_direction *= -1
                
                # During recovery: rotate in place with moderate speed
                v = 0.0
                w = self.recovery_direction * 0.9  # Slower recovery rotation
                trajectory = self.predict_trajectory(v, w)
        else:
            self.consecutive_low_velocity = 0
            # If we're moving well, cancel recovery
            if self.rotation_recovery_active and abs(v) > 0.15:
                rospy.loginfo("Movement restored - canceling recovery")
                self.rotation_recovery_active = False
                self.recovery_start_time = None
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        
        # Debug
        pursuit_time = (rospy.Time.now() - self.time_pursuing_goal).to_sec() if self.time_pursuing_goal else 0
        status = "RECOVERING" if self.rotation_recovery_active else "NORMAL"
        rospy.loginfo_throttle(1.0, 
            f"[{status}] v={v:.2f} w={w:.2f} | dist={dist_to_goal:.2f}m | t={pursuit_time:.0f}s")
        
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