#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
import tf2_ros
import tf.transformations as tf_trans

class CustomMapMerger:
    def __init__(self):
        rospy.init_node('custom_map_merger', anonymous=False)
        
        # Parameters
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.merged_map_topic = rospy.get_param('~merged_map_topic', 'merged_map')
        self.merge_rate = rospy.get_param('~merge_rate', 2.0)
        self.resolution = rospy.get_param('~resolution', 0.05)
        
        # Robot configurations
        self.robots = ['robot1', 'robot2', 'robot3']
        
        # Storage for received maps
        self.robot_maps = {}
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers for each robot's map
        self.subscribers = {}
        for robot in self.robots:
            topic = f'/{robot}/map'
            self.subscribers[robot] = rospy.Subscriber(
                topic, OccupancyGrid, self.map_callback, callback_args=robot
            )
            rospy.loginfo(f"Subscribed to {topic}")
        
        # Publisher for merged map
        self.merged_map_pub = rospy.Publisher(
            self.merged_map_topic, OccupancyGrid, queue_size=1, latch=True
        )
        
        # Timer for periodic merging
        self.merge_timer = rospy.Timer(
            rospy.Duration(1.0 / self.merge_rate), self.merge_and_publish
        )
        
        rospy.loginfo("Custom Map Merger initialized")
    
    def map_callback(self, msg, robot_name):
        """Store received map from robot"""
        self.robot_maps[robot_name] = msg
        rospy.loginfo_throttle(5.0, f"Received map from {robot_name}: {msg.info.width}x{msg.info.height}")
    
    def get_transform(self, target_frame, source_frame):
        """Get transform from TF tree"""
        try:
            # Get latest available transform
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            
            # Extract translation
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            
            # Extract rotation (yaw)
            quat = [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]
            euler = tf_trans.euler_from_quaternion(quat)
            yaw = euler[2]
            
            return (x, y, yaw)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"Could not get transform {target_frame} -> {source_frame}: {e}")
            return None
    
    def transform_point(self, x, y, tx, ty, theta):
        """Transform a point by translation (tx, ty) and rotation theta"""
        # Rotate
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        
        # Translate
        x_final = x_rot + tx
        y_final = y_rot + ty
        
        return x_final, y_final
    
    def merge_and_publish(self, event):
        """Merge all robot maps and publish"""
        if len(self.robot_maps) == 0:
            rospy.logwarn_throttle(5.0, "No maps received yet")
            return
        
        # Create merged map
        merged_map = self.create_merged_map()
        
        if merged_map is not None:
            self.merged_map_pub.publish(merged_map)
            rospy.loginfo_throttle(5.0, f"Published merged map: {merged_map.info.width}x{merged_map.info.height}")
    
    def create_merged_map(self):
        """Create merged occupancy grid from all robot maps using TF transforms"""
        if len(self.robot_maps) == 0:
            return None
        
        # Get transforms for all robots
        robot_transforms = {}
        for robot_name in self.robot_maps.keys():
            robot_map_frame = f"{robot_name}_map"
            transform = self.get_transform(self.world_frame, robot_map_frame)
            
            if transform is None:
                rospy.logwarn_throttle(5.0, f"No transform for {robot_name}, skipping")
                continue
            
            robot_transforms[robot_name] = transform
        
        if len(robot_transforms) == 0:
            rospy.logwarn_throttle(5.0, "No valid transforms found")
            return None
        
        # Calculate world bounds by transforming map corners
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for robot_name, robot_map in self.robot_maps.items():
            if robot_name not in robot_transforms:
                continue
            
            tx, ty, theta = robot_transforms[robot_name]
            
            # Get map origin in map frame
            origin_x = robot_map.info.origin.position.x
            origin_y = robot_map.info.origin.position.y
            
            # Get map dimensions
            width_m = robot_map.info.width * robot_map.info.resolution
            height_m = robot_map.info.height * robot_map.info.resolution
            
            # Four corners of the map in map frame
            corners = [
                (origin_x, origin_y),
                (origin_x + width_m, origin_y),
                (origin_x, origin_y + height_m),
                (origin_x + width_m, origin_y + height_m)
            ]
            
            # Transform corners to world frame
            for corner_x, corner_y in corners:
                world_x, world_y = self.transform_point(corner_x, corner_y, tx, ty, theta)
                min_x = min(min_x, world_x)
                max_x = max(max_x, world_x)
                min_y = min(min_y, world_y)
                max_y = max(max_y, world_y)
        
        # Add padding
        padding = 2.0
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Create merged map dimensions
        width = int(np.ceil((max_x - min_x) / self.resolution))
        height = int(np.ceil((max_y - min_y) / self.resolution))
        
        # Initialize merged grid
        merged_grid = np.full((height, width), -1, dtype=np.int8)
        
        # Merge each robot's map
        for robot_name, robot_map in self.robot_maps.items():
            if robot_name not in robot_transforms:
                continue
            
            tx, ty, theta = robot_transforms[robot_name]
            
            # Get robot map data
            robot_data = np.array(robot_map.data).reshape(
                (robot_map.info.height, robot_map.info.width)
            )
            
            # Map origin in map frame
            map_origin_x = robot_map.info.origin.position.x
            map_origin_y = robot_map.info.origin.position.y
            
            # Transform each cell
            for i in range(robot_map.info.height):
                for j in range(robot_map.info.width):
                    value = robot_data[i, j]
                    
                    if value == -1:  # Skip unknown cells
                        continue
                    
                    # Cell position in map frame
                    cell_x = map_origin_x + j * robot_map.info.resolution
                    cell_y = map_origin_y + i * robot_map.info.resolution
                    
                    # Transform to world frame
                    world_x, world_y = self.transform_point(cell_x, cell_y, tx, ty, theta)
                    
                    # Convert to merged grid indices
                    merged_j = int((world_x - min_x) / self.resolution)
                    merged_i = int((world_y - min_y) / self.resolution)
                    
                    # Check bounds
                    if 0 <= merged_i < height and 0 <= merged_j < width:
                        # Merge strategy: take max value (more conservative for obstacles)
                        if merged_grid[merged_i, merged_j] == -1:
                            merged_grid[merged_i, merged_j] = value
                        else:
                            merged_grid[merged_i, merged_j] = max(
                                merged_grid[merged_i, merged_j], value
                            )
        
        # Create OccupancyGrid message
        merged_msg = OccupancyGrid()
        merged_msg.header.stamp = rospy.Time.now()
        merged_msg.header.frame_id = self.world_frame
        
        merged_msg.info.resolution = self.resolution
        merged_msg.info.width = width
        merged_msg.info.height = height
        
        merged_msg.info.origin.position.x = min_x
        merged_msg.info.origin.position.y = min_y
        merged_msg.info.origin.position.z = 0.0
        merged_msg.info.origin.orientation.w = 1.0
        
        merged_msg.data = merged_grid.flatten().tolist()
        
        return merged_msg
    
    def run(self):
        """Keep node running"""
        rospy.spin()

if __name__ == '__main__':
    try:
        merger = CustomMapMerger()
        merger.run()
    except rospy.ROSInterruptException:
        pass
