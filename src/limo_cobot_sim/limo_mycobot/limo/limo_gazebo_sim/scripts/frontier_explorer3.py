#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from scipy.ndimage import label
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class FrontierExplorer:
    """Frontier-based exploration with smart retry logic"""
    def __init__(self):
        rospy.init_node('frontier_explorer')
        
        self.map_data = None
        self.map_info = None
        self.current_pose = None
        self.current_goal = None
        self.goal_active = False
        self.start_time = rospy.Time.now()
        self.initial_delay = 5.0
        
        # Track explored regions and failed goals
        self.visited_frontiers = []
        self.failed_goals = []
        self.permanently_failed_goals = []  # Goals that failed multiple times
        self.visit_radius = 1.5
        self.failed_goal_radius = 2.0
        self.exploration_history = []
        
        # Track retry attempts
        self.retry_attempts = {}  # Track how many times each location was tried
        
        # Track completion
        self.no_valid_frontier_count = 0
        self.max_no_frontier_attempts = 5
        self.exploration_complete = False
        
        # Track goal timeout
        self.goal_sent_time = None
        self.max_goal_wait = 45.0  # If no response in 45s, consider it failed
        
        # Coverage tracking
        self.explored_map = None
        
        # Parameters
        self.min_frontier_size = rospy.get_param('~min_frontier_size', 10)
        self.exploration_rate = rospy.get_param('~exploration_rate', 3.0)
        
        # Exploration strategy parameters
        self.distance_weight = 0.5
        self.info_gain_weight = 3.0
        self.size_weight = 1.0
        self.novelty_weight = 2.0
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/robot3/map', OccupancyGrid, self.map_callback)
        self.goal_reached_sub = rospy.Subscriber('/robot3/goal_reached', Bool, self.goal_reached_callback)
        self.goal_failed_sub = rospy.Subscriber('/robot3/goal_failed', Bool, self.goal_failed_callback)
        
        # Publishers
        self.goal_pub = rospy.Publisher('/robot3/goal', PoseStamped, queue_size=10)
        self.frontier_pub = rospy.Publisher('/robot3/frontiers', MarkerArray, queue_size=10)
        
        # Timer for exploration
        self.exploration_timer = rospy.Timer(rospy.Duration(self.exploration_rate), 
                                            self.exploration_callback)
        
        rospy.loginfo("Smart Frontier Explorer initialized with retry logic")
        
    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        if self.explored_map is None or self.explored_map.shape != self.map_data.shape:
            self.explored_map = np.zeros_like(self.map_data, dtype=bool)
        
        self.explored_map |= (self.map_data != -1)
        
    def goal_reached_callback(self, msg):
        """Handle successful goal"""
        if msg.data:
            rospy.loginfo("✅ Goal reached successfully")
            self.goal_active = False
            
            if self.current_goal is not None:
                self.visited_frontiers.append(
                    (self.current_goal.pose.position.x, 
                     self.current_goal.pose.position.y)
                )
                if len(self.visited_frontiers) > 20:
                    self.visited_frontiers.pop(0)
            
            self.current_goal = None
            self.goal_sent_time = None
    
    def goal_failed_callback(self, msg):
        """Handle failed goal with retry tracking"""
        if msg.data:
            rospy.logwarn("❌ robot3 Goal failed - will avoid this area")
            self.goal_active = False
            
            if self.current_goal is not None:
                goal_pos = (self.current_goal.pose.position.x,
                           self.current_goal.pose.position.y)
                
                # Track retry attempts
                goal_key = f"{goal_pos[0]:.1f},{goal_pos[1]:.1f}"
                if goal_key not in self.retry_attempts:
                    self.retry_attempts[goal_key] = 0
                self.retry_attempts[goal_key] += 1
                
                # If failed 3+ times, mark as permanently failed
                if self.retry_attempts[goal_key] >= 3:
                    if goal_pos not in self.permanently_failed_goals:
                        self.permanently_failed_goals.append(goal_pos)
                        rospy.logerr(f"🚫 Permanently failed (3+ attempts) at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
                else:
                    # Otherwise just add to regular failed list
                    self.failed_goals.append(goal_pos)
                    rospy.logwarn(f"   Failed goal at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}) "
                                f"(attempt {self.retry_attempts[goal_key]})")
                
                # Keep failed list manageable
                if len(self.failed_goals) > 15:
                    self.failed_goals.pop(0)
            
            self.current_goal = None
            self.goal_sent_time = None
        
    def get_robot_pose(self):
        """Get current robot pose"""
        try:
            trans = self.tf_buffer.lookup_transform('robot3_map', 'robot3_base_link', 
                                                   rospy.Time(0), rospy.Duration(1.0))
            pose = PoseStamped()
            pose.header.frame_id = 'robot3_map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.orientation = trans.transform.rotation
            
            pos = (trans.transform.translation.x, trans.transform.translation.y)
            if len(self.exploration_history) == 0 or \
               np.linalg.norm(np.array(pos) - np.array(self.exploration_history[-1])) > 0.3:
                self.exploration_history.append(pos)
                if len(self.exploration_history) > 100:
                    self.exploration_history.pop(0)
            
            return pose
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to get robot pose: {e}")
            return None
        
    def world_to_map(self, x, y):
        mx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return mx, my
        
    def map_to_world(self, mx, my):
        x = mx * self.map_info.resolution + self.map_info.origin.position.x
        y = my * self.map_info.resolution + self.map_info.origin.position.y
        return x, y
        
    def is_frontier_cell(self, x, y):
        if self.map_data[y, x] != -1:
            return False
        
        free_neighbors = 0
        occupied_neighbors = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.map_info.width and 
                    0 <= ny < self.map_info.height):
                    val = self.map_data[ny, nx]
                    if val == 0:
                        free_neighbors += 1
                    elif val > 50:
                        occupied_neighbors += 1
        
        return free_neighbors >= 1 and occupied_neighbors < 6
        
    def find_frontiers(self):
        if self.map_data is None:
            return []
            
        height, width = self.map_data.shape
        
        frontier_map = np.zeros((height, width), dtype=bool)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if self.is_frontier_cell(x, y):
                    frontier_map[y, x] = True
                    
        labeled_frontiers, num_frontiers = label(frontier_map, structure=np.ones((3, 3)))
        
        frontiers = []
        for i in range(1, num_frontiers + 1):
            frontier_cells = np.argwhere(labeled_frontiers == i)
            
            if len(frontier_cells) < self.min_frontier_size:
                continue
                
            centroid_y, centroid_x = np.mean(frontier_cells, axis=0)
            world_x, world_y = self.map_to_world(int(centroid_x), int(centroid_y))
            
            frontiers.append({
                'centroid': (world_x, world_y),
                'size': len(frontier_cells),
                'cells': frontier_cells
            })
            
        return frontiers
        
    def is_frontier_visited(self, frontier):
        fx, fy = frontier['centroid']
        
        for visited_x, visited_y in self.visited_frontiers:
            dist = np.sqrt((fx - visited_x)**2 + (fy - visited_y)**2)
            if dist < self.visit_radius:
                return True
        
        return False
    
    def is_near_failed_goal(self, frontier):
        """Check if frontier is near a failed goal"""
        fx, fy = frontier['centroid']
        
        # Check permanently failed goals first
        for failed_x, failed_y in self.permanently_failed_goals:
            dist = np.sqrt((fx - failed_x)**2 + (fy - failed_y)**2)
            if dist < self.failed_goal_radius:
                return True
        
        # Then check regular failed goals
        for failed_x, failed_y in self.failed_goals:
            dist = np.sqrt((fx - failed_x)**2 + (fy - failed_y)**2)
            if dist < self.failed_goal_radius:
                return True
        
        return False
    
    def calculate_novelty_score(self, frontier, robot_pose):
        fx, fy = frontier['centroid']
        
        if len(self.exploration_history) < 5:
            return 1.0
        
        min_dist = float('inf')
        for hist_x, hist_y in self.exploration_history:
            dist = np.sqrt((fx - hist_x)**2 + (fy - hist_y)**2)
            min_dist = min(min_dist, dist)
        
        novelty = 1.0 / (1.0 + np.exp(-0.5 * (min_dist - 2.0)))
        return novelty
        
    def calculate_information_gain(self, frontier, robot_pose):
        if self.map_data is None:
            return 0.0
        
        cx, cy = frontier['centroid']
        mx, my = self.world_to_map(cx, cy)
        
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y
        
        direction_angle = np.arctan2(cy - ry, cx - rx)
        
        search_radius = 40
        unknown_count = 0
        total_count = 0
        known_free_count = 0
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                dist = np.sqrt(dx**2 + dy**2)
                if dist > search_radius or dist < 3:
                    continue
                
                nx, ny = mx + dx, my + dy
                if (0 <= nx < self.map_info.width and 
                    0 <= ny < self.map_info.height):
                    
                    angle_to_cell = np.arctan2(dy, dx)
                    angle_diff = abs(np.arctan2(np.sin(angle_to_cell - direction_angle), 
                                                 np.cos(angle_to_cell - direction_angle)))
                    
                    if angle_diff < np.pi / 1.5:
                        weight = 1.0 if angle_diff < np.pi / 3 else 0.5
                        total_count += weight
                        
                        cell_val = self.map_data[ny, nx]
                        if cell_val == -1:
                            unknown_count += weight
                        elif cell_val == 0:
                            known_free_count += weight
        
        if total_count < 10:
            return 0.0
        
        info_gain = unknown_count / total_count
        
        free_ratio = known_free_count / total_count
        if free_ratio > 0.7:
            info_gain *= 0.3
        
        size_boost = min(1.5, 1.0 + frontier['size'] / 100.0)
        
        return info_gain * size_boost
        
    def calculate_frontier_cost(self, frontier, robot_pose):
        fx, fy = frontier['centroid']
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y
        
        distance = np.sqrt((fx - rx)**2 + (fy - ry)**2)
        
        info_gain = self.calculate_information_gain(frontier, robot_pose)
        novelty = self.calculate_novelty_score(frontier, robot_pose)
        
        if info_gain < 0.1:
            return float('inf')
        
        max_reasonable_dist = 8.0
        normalized_dist = min(1.0, distance / max_reasonable_dist)
        normalized_size = min(1.0, frontier['size'] / 100.0)
        
        cost = (
            self.distance_weight * normalized_dist +
            self.info_gain_weight * (1.0 - info_gain) +
            self.size_weight * (1.0 - normalized_size) +
            self.novelty_weight * (1.0 - novelty)
        )
        
        return cost
        
    def select_best_frontier(self, frontiers, robot_pose):
        if len(frontiers) == 0:
            return None
        
        frontier_info = []
        for frontier in frontiers:
            info_gain = self.calculate_information_gain(frontier, robot_pose)
            novelty = self.calculate_novelty_score(frontier, robot_pose)
            cost = self.calculate_frontier_cost(frontier, robot_pose)
            
            if cost == float('inf'):
                continue
            
            distance = np.sqrt(
                (frontier['centroid'][0] - robot_pose.pose.position.x)**2 + 
                (frontier['centroid'][1] - robot_pose.pose.position.y)**2
            )
            
            frontier_info.append({
                'frontier': frontier,
                'cost': cost,
                'info_gain': info_gain,
                'novelty': novelty,
                'distance': distance
            })
        
        if len(frontier_info) == 0:
            return None
        
        frontier_info.sort(key=lambda x: x['cost'])
        
        rospy.loginfo("="*60)
        rospy.loginfo("Top frontier candidates:")
        for i, finfo in enumerate(frontier_info[:3]):
            rospy.loginfo(
                f"  #{i+1}: dist={finfo['distance']:.2f}m, "
                f"size={finfo['frontier']['size']}, "
                f"info={finfo['info_gain']:.2f}"
            )
        rospy.loginfo("="*60)
        
        return frontier_info[0]['frontier']
        
    def get_approach_point(self, frontier):
        cx, cy = frontier['centroid']
        mx, my = self.world_to_map(cx, cy)
        
        best_x, best_y = mx, my
        best_dist = float('inf')
        
        search_radius = 20
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = mx + dx, my + dy
                
                if (0 <= nx < self.map_info.width and 
                    0 <= ny < self.map_info.height):
                    
                    if self.map_data[ny, nx] == 0:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_x, best_y = nx, ny
        
        return self.map_to_world(best_x, best_y)
        
    def publish_frontiers_markers(self, frontiers, robot_pose):
        marker_array = MarkerArray()
        
        for i, frontier in enumerate(frontiers):
            info_gain = self.calculate_information_gain(frontier, robot_pose)
            novelty = self.calculate_novelty_score(frontier, robot_pose)
            
            marker = Marker()
            marker.header.frame_id = "robot3_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = frontier['centroid'][0]
            marker.pose.position.y = frontier['centroid'][1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            value_score = info_gain * novelty
            scale = 0.2 + 0.3 * min(1.0, value_score)
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            marker.color.r = 1.0 - value_score
            marker.color.g = value_score
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
        self.frontier_pub.publish(marker_array)
        
    def check_exploration_complete(self):
        if self.map_data is None:
            return False
        
        total_cells = self.map_data.size
        unknown_cells = np.sum(self.map_data == -1)
        unknown_ratio = unknown_cells / total_cells
        
        return unknown_ratio < 0.03
    
    def clear_failed_goals_for_frontiers(self, frontiers):
        """Clear ONLY regular failed goals (not permanent) when retrying frontiers"""
        if len(frontiers) == 0 or len(self.failed_goals) == 0:
            return
        
        cleared_count = 0
        new_failed_goals = []
        
        for failed_x, failed_y in self.failed_goals:
            should_keep = True
            
            # Check if this failed goal blocks an available frontier
            for frontier in frontiers:
                fx, fy = frontier['centroid']
                dist = np.sqrt((fx - failed_x)**2 + (fy - failed_y)**2)
                
                if dist < self.failed_goal_radius:
                    # Clear this failed goal to give it another chance
                    should_keep = False
                    cleared_count += 1
                    rospy.loginfo(f"🔓 Clearing failed goal at ({failed_x:.2f}, {failed_y:.2f}) "
                                f"to retry frontier at ({fx:.2f}, {fy:.2f})")
                    break
            
            if should_keep:
                new_failed_goals.append((failed_x, failed_y))
        
        if cleared_count > 0:
            self.failed_goals = new_failed_goals
            rospy.loginfo(f"✨ Cleared {cleared_count} failed goals to retry available frontiers")
            rospy.loginfo(f"   ({len(self.permanently_failed_goals)} remain permanently failed after 3+ attempts)")
        
    def exploration_callback(self, event):
        """Main exploration loop"""
        # Wait for initial map
        if (rospy.Time.now() - self.start_time).to_sec() < self.initial_delay:
            remaining = self.initial_delay - (rospy.Time.now() - self.start_time).to_sec()
            rospy.loginfo_throttle(2.0, f"Building initial map... {remaining:.1f}s")
            return
        
        if self.exploration_complete:
            rospy.loginfo_throttle(10.0, "🎉 Exploration COMPLETE!")
            return
        
        if self.map_data is None:
            return
            
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        
        # CHECK: Has goal timed out without response?
        if self.goal_active and self.goal_sent_time is not None:
            wait_time = (rospy.Time.now() - self.goal_sent_time).to_sec()
            if wait_time > self.max_goal_wait:
                rospy.logerr(f"⏱️ Goal timeout - no response in {wait_time:.1f}s, assuming failed")
                self.goal_active = False
                
                # Mark as failed
                if self.current_goal is not None:
                    goal_pos = (self.current_goal.pose.position.x,
                               self.current_goal.pose.position.y)
                    goal_key = f"{goal_pos[0]:.1f},{goal_pos[1]:.1f}"
                    
                    if goal_key not in self.retry_attempts:
                        self.retry_attempts[goal_key] = 0
                    self.retry_attempts[goal_key] += 1
                    
                    if self.retry_attempts[goal_key] >= 3:
                        self.permanently_failed_goals.append(goal_pos)
                    else:
                        self.failed_goals.append(goal_pos)
                
                self.current_goal = None
                self.goal_sent_time = None
        
        # Wait if goal still active
        if self.goal_active:
            return
            
        # Find frontiers
        frontiers = self.find_frontiers()
        
        if len(frontiers) == 0:
            self.no_valid_frontier_count += 1
            
            if self.no_valid_frontier_count >= self.max_no_frontier_attempts:
                if self.check_exploration_complete():
                    rospy.loginfo("="*70)
                    rospy.loginfo("🎉 EXPLORATION COMPLETE!")
                    rospy.loginfo("="*70)
                    self.exploration_complete = True
                else:
                    rospy.logwarn("Clearing visited history")
                    self.visited_frontiers.clear()
                    self.no_valid_frontier_count = 0
            return
        
        self.no_valid_frontier_count = 0
        
        # Filter frontiers - first pass (only check visited)
        unvisited = [f for f in frontiers if not self.is_frontier_visited(f)]
        
        # Check how many would be blocked by failed goals
        available_after_failed_filter = [f for f in unvisited if not self.is_near_failed_goal(f)]
        blocked_by_failed = [f for f in unvisited if self.is_near_failed_goal(f)]
        
        # Debug: Show what's happening
        rospy.loginfo_throttle(5.0, f"Filter status: {len(unvisited)} unvisited, "
                              f"{len(available_after_failed_filter)} after failed filter, "
                              f"{len(blocked_by_failed)} blocked by failed goals, "
                              f"{len(self.permanently_failed_goals)} permanently failed")
        
        # KEY LOGIC: If we have few available frontiers and some are blocked by failed goals, clear them
        if len(blocked_by_failed) > 0 and len(available_after_failed_filter) <= 3:
            rospy.logwarn("="*70)
            rospy.logwarn(f"📍robot3 Only {len(available_after_failed_filter)} available frontiers, "
                         f"but {len(blocked_by_failed)} more are blocked by failed goals!")
            rospy.logwarn(f"🔄 Clearing {len(blocked_by_failed)} failed goals to give them another try...")
            rospy.logwarn("   (They will be marked failed again only if they fail after retry)")
            rospy.logwarn("="*70)
            
            # Clear the failed goals that block these frontiers
            self.clear_failed_goals_for_frontiers(blocked_by_failed)
            
            # Recalculate available frontiers
            available_after_failed_filter = [f for f in unvisited if not self.is_near_failed_goal(f)]
            rospy.loginfo(f"✅ After clearing: {len(available_after_failed_filter)} frontiers now available")
        
        unvisited = available_after_failed_filter
        
        # If still no unvisited, clear visited history
        if len(unvisited) == 0:
            rospy.logwarn(f"All frontiers visited - clearing history")
            self.visited_frontiers.clear()
            unvisited = [f for f in frontiers if not self.is_near_failed_goal(f)]
            
            # If STILL nothing, clear failed goals too
            if len(unvisited) == 0:
                rospy.logwarn("All near failed goals - clearing failed list completely")
                self.failed_goals.clear()
                unvisited = [f for f in frontiers if not self.is_near_permanently_failed(f)]
            
        rospy.loginfo(f"📊 Frontiers: {len(frontiers)} total, {len(unvisited)} available after filtering, "
                     f"{len(self.failed_goals)} failed areas (retryable), "
                     f"{len(self.permanently_failed_goals)} permanently failed (3+ attempts), "
                     f"{len(self.visited_frontiers)} visited")
        
        # ADDITIONAL CHECK: If we keep seeing the same low number, be more aggressive
        if not hasattr(self, 'low_frontier_count'):
            self.low_frontier_count = 0
            self.last_available_count = 0
        
        if len(unvisited) > 0 and len(unvisited) <= 3 and len(self.failed_goals) > 3:
            if len(unvisited) == self.last_available_count:
                self.low_frontier_count += 1
            else:
                self.low_frontier_count = 0
            self.last_available_count = len(unvisited)
            
            # If stuck with few options for multiple cycles, clear oldest failed goals
            if self.low_frontier_count >= 3:
                rospy.logwarn("="*70)
                rospy.logwarn(f"⚠️ Stuck with only {len(unvisited)} frontiers for {self.low_frontier_count} cycles")
                rospy.logwarn(f"🧹 Clearing oldest 50% of failed goals to free up options")
                rospy.logwarn("="*70)
                
                # Keep only the most recent half of failed goals
                keep_count = max(2, len(self.failed_goals) // 2)
                old_count = len(self.failed_goals)
                self.failed_goals = self.failed_goals[-keep_count:]
                rospy.loginfo(f"Reduced failed goals from {old_count} to {len(self.failed_goals)}")
                
                # Recalculate available
                unvisited = [f for f in frontiers if not self.is_frontier_visited(f) 
                            and not self.is_near_failed_goal(f)]
                rospy.loginfo(f"✅ Now have {len(unvisited)} available frontiers")
                
                self.low_frontier_count = 0
        else:
            self.low_frontier_count = 0
        
        # Publish markers
        self.publish_frontiers_markers(unvisited, robot_pose)
        
        # Select best
        best_frontier = self.select_best_frontier(unvisited, robot_pose)
        
        if best_frontier is None:
            self.no_valid_frontier_count += 1
            
            if self.no_valid_frontier_count >= self.max_no_frontier_attempts:
                if self.check_exploration_complete():
                    rospy.loginfo("="*70)
                    rospy.loginfo("🎉 EXPLORATION COMPLETE!")
                    rospy.loginfo("="*70)
                    self.exploration_complete = True
            return
        
        self.no_valid_frontier_count = 0
            
        # Create and send goal
        approach_x, approach_y = self.get_approach_point(best_frontier)
        
        goal = PoseStamped()
        goal.header.frame_id = "robot3_map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = approach_x
        goal.pose.position.y = approach_y
        goal.pose.orientation.w = 1.0
        
        self.current_goal = goal
        self.goal_active = True
        self.goal_sent_time = rospy.Time.now()  # Track when we sent it
        self.goal_pub.publish(goal)
        
        rospy.loginfo(f"🎯 NEW GOAL: ({approach_x:.2f}, {approach_y:.2f})")
    
    def is_near_permanently_failed(self, frontier):
        """Check only permanently failed goals"""
        fx, fy = frontier['centroid']
        
        for failed_x, failed_y in self.permanently_failed_goals:
            dist = np.sqrt((fx - failed_x)**2 + (fy - failed_y)**2)
            if dist < self.failed_goal_radius:
                return True
        
        return False
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        explorer = FrontierExplorer()
        explorer.run()
    except rospy.ROSInterruptException:
        pass