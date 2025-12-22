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
    """Frontier-based exploration with aggressive recovery from stuck states"""
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
        self.permanently_failed_goals = []
        self.visit_radius = 1.5
        self.failed_goal_radius = 2.5  # Larger radius to avoid problem areas
        self.exploration_history = []
        
        # Track retry attempts
        self.retry_attempts = {}
        
        # Track completion
        self.no_valid_frontier_count = 0
        self.max_no_frontier_attempts = 3  # Faster reset
        self.exploration_complete = False
        
        # Track goal timeout
        self.goal_sent_time = None
        self.max_goal_wait = 50.0  # Shorter timeout
        
        # Stuck detection - more aggressive
        self.stuck_cycle_count = 0
        self.last_frontier_count = 0
        self.last_available_count = 0
        
        # Coverage tracking
        self.explored_map = None
        
        # Parameters
        self.min_frontier_size = rospy.get_param('~min_frontier_size', 10)
        self.exploration_rate = rospy.get_param('~exploration_rate', 3.0)
        
        # More aggressive exploration strategy
        self.distance_weight = 0.4  # Less weight on distance
        self.info_gain_weight = 3.5  # Higher info gain
        self.size_weight = 1.2
        self.novelty_weight = 2.5  # Much higher novelty preference
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers - NO NAMESPACES
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.goal_reached_sub = rospy.Subscriber('/goal_reached', Bool, self.goal_reached_callback)
        self.goal_failed_sub = rospy.Subscriber('/goal_failed', Bool, self.goal_failed_callback)
        
        # Publishers - NO NAMESPACES
        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=10)
        self.frontier_pub = rospy.Publisher('/frontiers', MarkerArray, queue_size=10)
        
        # Timer
        self.exploration_timer = rospy.Timer(rospy.Duration(self.exploration_rate), 
                                            self.exploration_callback)
        
        rospy.loginfo("Improved Frontier Explorer - Aggressive stuck recovery")
        
    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        if self.explored_map is None or self.explored_map.shape != self.map_data.shape:
            self.explored_map = np.zeros_like(self.map_data, dtype=bool)
        
        self.explored_map |= (self.map_data != -1)
        
    def goal_reached_callback(self, msg):
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
            self.stuck_cycle_count = 0  # Reset on success
    
    def goal_failed_callback(self, msg):
        if msg.data:
            rospy.logwarn("❌ Goal failed")
            self.goal_active = False
            
            if self.current_goal is not None:
                goal_pos = (self.current_goal.pose.position.x,
                           self.current_goal.pose.position.y)
                
                goal_key = f"{goal_pos[0]:.1f},{goal_pos[1]:.1f}"
                if goal_key not in self.retry_attempts:
                    self.retry_attempts[goal_key] = 0
                self.retry_attempts[goal_key] += 1
                
                # Faster permanent failure - 2 attempts instead of 3
                if self.retry_attempts[goal_key] >= 2:
                    if goal_pos not in self.permanently_failed_goals:
                        self.permanently_failed_goals.append(goal_pos)
                        rospy.logerr(f"🚫 Permanently failed at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
                else:
                    self.failed_goals.append(goal_pos)
                    rospy.logwarn(f"Failed attempt {self.retry_attempts[goal_key]}")
                
                if len(self.failed_goals) > 10:
                    self.failed_goals.pop(0)
            
            self.current_goal = None
            self.goal_sent_time = None
        
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', 
                                                   rospy.Time(0), rospy.Duration(1.0))
            pose = PoseStamped()
            pose.header.frame_id = 'map'
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
        fx, fy = frontier['centroid']
        
        for failed_x, failed_y in self.permanently_failed_goals:
            dist = np.sqrt((fx - failed_x)**2 + (fy - failed_y)**2)
            if dist < self.failed_goal_radius:
                return True
        
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
        
        # More aggressive novelty scoring
        novelty = 1.0 / (1.0 + np.exp(-0.8 * (min_dist - 1.5)))
        return novelty
        
    def calculate_information_gain(self, frontier, robot_pose):
        if self.map_data is None:
            return 0.0
        
        cx, cy = frontier['centroid']
        mx, my = self.world_to_map(cx, cy)
        
        search_radius = 40
        unknown_count = 0
        total_count = 0
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                dist = np.sqrt(dx**2 + dy**2)
                if dist > search_radius or dist < 3:
                    continue
                
                nx, ny = mx + dx, my + dy
                if (0 <= nx < self.map_info.width and 
                    0 <= ny < self.map_info.height):
                    total_count += 1
                    if self.map_data[ny, nx] == -1:
                        unknown_count += 1
        
        if total_count < 10:
            return 0.0
        
        info_gain = unknown_count / total_count
        size_boost = min(1.5, 1.0 + frontier['size'] / 100.0)
        
        return info_gain * size_boost
        
    def calculate_frontier_cost(self, frontier, robot_pose):
        fx, fy = frontier['centroid']
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y
        
        distance = np.sqrt((fx - rx)**2 + (fy - ry)**2)
        info_gain = self.calculate_information_gain(frontier, robot_pose)
        novelty = self.calculate_novelty_score(frontier, robot_pose)
        
        if info_gain < 0.08:
            return float('inf')
        
        max_reasonable_dist = 10.0
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
                f"novelty={finfo['novelty']:.2f}, "
                f"info={finfo['info_gain']:.2f}"
            )
        rospy.loginfo("="*60)
        
        return frontier_info[0]['frontier']
        
    def get_approach_point(self, frontier):
        cx, cy = frontier['centroid']
        mx, my = self.world_to_map(cx, cy)
        
        best_x, best_y = mx, my
        best_dist = float('inf')
        
        search_radius = 25  # Larger search for safer approach
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
            marker = Marker()
            marker.header.frame_id = "map"  # Changed from robot1_map
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = frontier['centroid'][0]
            marker.pose.position.y = frontier['centroid'][1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            marker.color.r = 0.0
            marker.color.g = 1.0
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
    
    def exploration_callback(self, event):
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
        
        # CHECK GOAL TIMEOUT
        if self.goal_active and self.goal_sent_time is not None:
            wait_time = (rospy.Time.now() - self.goal_sent_time).to_sec()
            if wait_time > self.max_goal_wait:
                rospy.logwarn(f"⏱️ Goal timeout (backup check)")
                self.goal_active = False
                
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
        
        if self.goal_active:
            return
            
        # FIND FRONTIERS
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
                    rospy.logwarn("No frontiers - clearing all history")
                    self.visited_frontiers.clear()
                    self.failed_goals.clear()
                    self.no_valid_frontier_count = 0
            return
        
        self.no_valid_frontier_count = 0
        
        # FILTER FRONTIERS
        unvisited = [f for f in frontiers if not self.is_frontier_visited(f)]
        available = [f for f in unvisited if not self.is_near_failed_goal(f)]
        
        # DETECT STUCK STATE
        if len(available) == self.last_available_count and len(frontiers) == self.last_frontier_count:
            self.stuck_cycle_count += 1
        else:
            self.stuck_cycle_count = 0
        
        self.last_available_count = len(available)
        self.last_frontier_count = len(frontiers)
        
        # AGGRESSIVE RECOVERY FROM STUCK STATE
        if self.stuck_cycle_count >= 2:  # Stuck for 2 cycles (6 seconds)
            rospy.logerr("="*70)
            rospy.logerr(f"🚨 STUCK DETECTED - Same {len(available)} frontiers for {self.stuck_cycle_count} cycles!")
            rospy.logerr("💥 NUCLEAR RESET - Clearing ALL failed goals")
            rospy.logerr("="*70)
            
            # Clear everything except permanent failures
            self.failed_goals.clear()
            self.visited_frontiers.clear()
            self.stuck_cycle_count = 0
            self.retry_attempts.clear()
            
            # Recalculate available
            available = [f for f in unvisited if not self.is_near_permanently_failed(f)]
            rospy.logwarn(f"✅ After nuclear reset: {len(available)} frontiers now available")
        
        # If still very few options, clear visited too
        if len(available) <= 2 and len(unvisited) > 3:
            rospy.logwarn("Very few available - clearing visited history")
            self.visited_frontiers.clear()
            available = [f for f in frontiers if not self.is_near_failed_goal(f)]
        
        # Last resort - ignore all failures
        if len(available) == 0:
            rospy.logerr("NO AVAILABLE FRONTIERS - Ignoring all failures!")
            self.failed_goals.clear()
            available = [f for f in unvisited if not self.is_near_permanently_failed(f)]
            
            if len(available) == 0:
                rospy.logerr("Still nothing - clearing permanent failures too!")
                self.permanently_failed_goals.clear()
                available = frontiers
        
        rospy.loginfo(f"📊 Frontiers: {len(frontiers)} total, {len(available)} available after filtering, "
                     f"{len(self.failed_goals)} failed areas (retryable), "
                     f"{len(self.permanently_failed_goals)} permanently failed (3+ attempts), "
                     f"{len(self.visited_frontiers)} visited")
        
        self.publish_frontiers_markers(available, robot_pose)
        
        # SELECT BEST
        best_frontier = self.select_best_frontier(available, robot_pose)
        
        if best_frontier is None:
            self.no_valid_frontier_count += 1
            if self.no_valid_frontier_count >= self.max_no_frontier_attempts:
                if self.check_exploration_complete():
                    self.exploration_complete = True
            return
        
        self.no_valid_frontier_count = 0
            
        # SEND GOAL
        approach_x, approach_y = self.get_approach_point(best_frontier)
        
        goal = PoseStamped()
        goal.header.frame_id = "map"  # Changed from robot1_map
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = approach_x
        goal.pose.position.y = approach_y
        goal.pose.orientation.w = 1.0
        
        self.current_goal = goal
        self.goal_active = True
        self.goal_sent_time = rospy.Time.now()
        self.goal_pub.publish(goal)
        
        rospy.loginfo(f"🎯 NEW GOAL: ({approach_x:.2f}, {approach_y:.2f})")
    
    def is_near_permanently_failed(self, frontier):
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