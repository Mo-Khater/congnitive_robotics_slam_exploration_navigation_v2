#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from scipy.ndimage import distance_transform_edt

class CostmapGenerator:
    """Generate costmap from occupancy grid and laser scan"""
    def __init__(self):
        rospy.init_node('costmap_generator')
        
        # Parameters
        self.inflation_radius = rospy.get_param('~inflation_radius', 0.5)
        self.cost_scaling_factor = rospy.get_param('~cost_scaling_factor', 10.0)
        self.lethal_cost = 100
        self.inscribed_radius = rospy.get_param('~inscribed_radius', 0.2)
        
        self.map_data = None
        self.map_info = None
        self.scan_data = None
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/robot2/map', OccupancyGrid, self.map_callback)
        self.scan_sub = rospy.Subscriber('/robot2/scan', LaserScan, self.scan_callback)
        
        # Publishers
        self.costmap_pub = rospy.Publisher('/robot2/costmap', OccupancyGrid, queue_size=10)
        
        # Timer for costmap generation
        self.timer = rospy.Timer(rospy.Duration(0.5), self.generate_costmap)
        
        rospy.loginfo("Costmap Generator initialized")
        
    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
    def scan_callback(self, msg):
        self.scan_data = msg
        
    def inflate_obstacles(self, grid):
        """Inflate obstacles based on robot radius"""
        if grid is None:
            return None
            
        # Create binary obstacle map
        obstacle_map = (grid > 50) | (grid == -1)
        
        # Calculate distance transform
        distance_map = distance_transform_edt(~obstacle_map)
        
        # Convert distances to costs
        costmap = np.zeros_like(grid, dtype=np.int8)
        
        # Calculate inflation radius in cells
        inflation_cells = int(self.inflation_radius / self.map_info.resolution)
        inscribed_cells = int(self.inscribed_radius / self.map_info.resolution)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                dist_cells = distance_map[i, j]
                
                if obstacle_map[i, j]:
                    # Lethal obstacle
                    costmap[i, j] = self.lethal_cost
                elif dist_cells <= inscribed_cells:
                    # Inside inscribed radius
                    costmap[i, j] = self.lethal_cost
                elif dist_cells <= inflation_cells:
                    # Inside inflation radius
                    # Cost decreases exponentially with distance
                    factor = (inflation_cells - dist_cells) / (inflation_cells - inscribed_cells)
                    cost = int(self.lethal_cost * np.exp(-self.cost_scaling_factor * factor))
                    costmap[i, j] = max(0, min(99, cost))
                else:
                    # Free space
                    if grid[i, j] == 0:
                        costmap[i, j] = 0
                    else:
                        costmap[i, j] = -1  # Unknown
                        
        return costmap
        
    def add_scan_obstacles(self, costmap):
        """Add obstacles from laser scan to costmap"""
        if self.scan_data is None or costmap is None:
            return costmap
            
        # This is a simplified version - in practice you'd want to use TF
        # to transform scan points to map frame
        
        return costmap
        
    def generate_costmap(self, event):
        """Generate and publish costmap"""
        if self.map_data is None or self.map_info is None:
            return
            
        # Inflate obstacles
        costmap = self.inflate_obstacles(self.map_data)
        
        if costmap is None:
            return
            
        # Add scan obstacles
        costmap = self.add_scan_obstacles(costmap)
        
        # Publish costmap
        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = rospy.Time.now()
        costmap_msg.header.frame_id = "robot2_map"
        costmap_msg.info = self.map_info
        costmap_msg.data = costmap.flatten().tolist()
        
        self.costmap_pub.publish(costmap_msg)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        generator = CostmapGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass