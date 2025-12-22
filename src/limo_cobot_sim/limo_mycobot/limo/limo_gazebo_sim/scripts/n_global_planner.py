#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import heapq
from scipy.ndimage import distance_transform_edt

class AStarPlanner:
    def __init__(self):
        rospy.init_node('global_planner')
        
        self.map_data = None
        self.map_info = None
        self.goal = None
        self.start = None
        self.distance_map = None  # Distance from obstacles
        
        # Subscribers - NO NAMESPACES
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.goal_sub = rospy.Subscriber('/goal', PoseStamped, self.goal_callback)
        self.pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.pose_callback)
        
        # Publishers - NO NAMESPACES
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=10)
        
        self.obstacle_threshold = 50  # Occupancy values > this are obstacles
        self.unknown_is_free = True  # Treat unknown space as free for initial movement
        
        # *** NEW PARAMETERS FOR OBSTACLE AVOIDANCE ***
        self.inflation_radius = 5  # cells to inflate obstacles (increase for more clearance)
        self.distance_weight = 10.0  # Weight for distance cost (higher = stay further from obstacles)
        
        rospy.loginfo("Global Planner initialized with obstacle inflation")
        
    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        # Compute distance transform when map is updated
        self.compute_distance_map()
        
    def compute_distance_map(self):
        """Compute distance from each cell to nearest obstacle"""
        # Create binary obstacle map (1 = free, 0 = obstacle)
        free_space = np.ones_like(self.map_data, dtype=bool)
        free_space[self.map_data > self.obstacle_threshold] = False
        free_space[self.map_data == -1] = True  # Treat unknown as free
        
        # Compute Euclidean distance transform
        self.distance_map = distance_transform_edt(free_space)
        rospy.loginfo("Distance map computed")
        
    def pose_callback(self, msg):
        self.start = msg
        
    def goal_callback(self, msg):
        self.goal = msg
        if self.map_data is not None and self.start is not None:
            self.plan_path()
            
    def world_to_map(self, x, y):
        """Convert world coordinates to map indices"""
        mx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return mx, my
        
    def map_to_world(self, mx, my):
        """Convert map indices to world coordinates"""
        x = mx * self.map_info.resolution + self.map_info.origin.position.x
        y = my * self.map_info.resolution + self.map_info.origin.position.y
        return x, y
        
    def is_valid(self, mx, my, allow_unknown=True, check_inflation=True):
        """Check if map coordinates are valid and not occupied"""
        if mx < 0 or mx >= self.map_info.width or my < 0 or my >= self.map_info.height:
            return False
        
        cell_value = self.map_data[my, mx]
        
        # Occupied cell
        if cell_value > self.obstacle_threshold:
            return False
        
        # Unknown cell
        if cell_value == -1:
            return allow_unknown  # Allow unknown cells if specified
        
        # *** NEW: Check if within inflation radius of obstacles ***
        if check_inflation and self.distance_map is not None:
            if self.distance_map[my, mx] < self.inflation_radius:
                return False
        
        # Free cell
        return True
    
    def find_nearest_free_cell(self, mx, my, max_radius=10):
        """Find nearest free cell from given position"""
        if self.is_valid(mx, my, allow_unknown=True, check_inflation=False):
            return mx, my
        
        # Search in expanding circles
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        nx, ny = mx + dx, my + dy
                        if self.is_valid(nx, ny, allow_unknown=True, check_inflation=False):
                            rospy.loginfo(f"Adjusted position from ({mx}, {my}) to ({nx}, {ny})")
                            return nx, ny
        
        return None, None
        
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_distance_cost(self, node):
        """Get cost based on distance from obstacles (lower distance = higher cost)"""
        if self.distance_map is None:
            return 0.0
        
        mx, my = node
        dist = self.distance_map[my, mx]
        
        # Exponential decay: closer to obstacles = higher cost
        # Using max to avoid division by zero
        if dist < 1.0:
            return self.distance_weight * 10.0  # Very high cost near obstacles
        elif dist < self.inflation_radius * 2:
            # Smooth cost gradient
            return self.distance_weight / dist
        else:
            return 0.0  # No penalty far from obstacles
        
    def get_neighbors(self, node):
        """Get valid neighbors (8-connected)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = node[0] + dx, node[1] + dy
                # Allow planning through unknown space but not inflated obstacles
                if self.is_valid(nx, ny, allow_unknown=True, check_inflation=False):
                    # Diagonal moves cost more
                    base_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    
                    # *** NEW: Add distance-based cost ***
                    distance_cost = self.get_distance_cost((nx, ny))
                    total_cost = base_cost + distance_cost
                    
                    neighbors.append(((nx, ny), total_cost))
        return neighbors
        
    def a_star(self, start, goal):
        """A* pathfinding algorithm with obstacle avoidance"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
                
            closed_set.add(current)
            
            for neighbor, cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None  # No path found
        
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from A* came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
        
    def plan_path(self):
        """Plan path from start to goal"""
        if self.distance_map is None:
            rospy.logwarn("Distance map not ready yet")
            return
            
        start_x = self.start.pose.position.x
        start_y = self.start.pose.position.y
        goal_x = self.goal.pose.position.x
        goal_y = self.goal.pose.position.y
        
        start_map = self.world_to_map(start_x, start_y)
        goal_map = self.world_to_map(goal_x, goal_y)
        
        # Find nearest free cells if start/goal are invalid
        start_map = self.find_nearest_free_cell(*start_map)
        if start_map[0] is None:
            rospy.logwarn("Cannot find valid start position nearby")
            return
        
        goal_map = self.find_nearest_free_cell(*goal_map)
        if goal_map[0] is None:
            rospy.logwarn("Cannot find valid goal position nearby")
            return
            
        rospy.loginfo(f"Planning path from {start_map} to {goal_map}")
        path_map = self.a_star(start_map, goal_map)
        
        if path_map is None:
            rospy.logwarn("No path found!")
            return
            
        # Convert to Path message
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"  # Changed from robot1_map
        
        for mx, my in path_map:
            x, y = self.map_to_world(mx, my)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)
        rospy.loginfo(f"Published path with {len(path_msg.poses)} waypoints")
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        planner = AStarPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass