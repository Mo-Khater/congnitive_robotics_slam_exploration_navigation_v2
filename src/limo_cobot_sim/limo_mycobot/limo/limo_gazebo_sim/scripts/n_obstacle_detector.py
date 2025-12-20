#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

class ObstacleDetector:
    """Detect and cluster obstacles from laser scan data"""
    def __init__(self):
        rospy.init_node('obstacle_detector')
        
        # Parameters
        self.cluster_distance_threshold = rospy.get_param('~cluster_distance', 0.3)
        self.min_cluster_size = rospy.get_param('~min_cluster_size', 3)
        self.max_detection_range = rospy.get_param('~max_range', 5.0)
        
        self.scan_data = None
        
        # Subscribers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Publishers
        self.obstacle_pub = rospy.Publisher('/obstacles', MarkerArray, queue_size=10)
        
        rospy.loginfo("Obstacle Detector initialized")
        
    def scan_callback(self, msg):
        self.scan_data = msg
        obstacles = self.detect_obstacles(msg)
        self.publish_obstacles(obstacles)
        
    def detect_obstacles(self, scan):
        """Detect obstacles from laser scan using clustering"""
        points = []
        
        # Convert scan to cartesian points
        for i, r in enumerate(scan.ranges):
            if r < scan.range_min or r > scan.range_max or r > self.max_detection_range:
                continue
            if np.isnan(r) or np.isinf(r):
                continue
                
            angle = scan.angle_min + i * scan.angle_increment
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
            
        if len(points) == 0:
            return []
            
        points = np.array(points)
        
        # Cluster points
        clusters = self.cluster_points(points)
        
        # Calculate obstacle centers and sizes
        obstacles = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
                
            center = np.mean(cluster, axis=0)
            size = np.max(np.linalg.norm(cluster - center, axis=1))
            
            obstacles.append({
                'center': center,
                'radius': size,
                'points': cluster
            })
            
        return obstacles
        
    def cluster_points(self, points):
        """Simple distance-based clustering"""
        if len(points) == 0:
            return []
            
        clusters = []
        used = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if used[i]:
                continue
                
            cluster = [points[i]]
            used[i] = True
            
            # Find nearby points
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                    
                # Check distance to any point in cluster
                min_dist = min([np.linalg.norm(points[j] - p) for p in cluster])
                
                if min_dist < self.cluster_distance_threshold:
                    cluster.append(points[j])
                    used[j] = True
                    
            clusters.append(np.array(cluster))
            
        return clusters
        
    def publish_obstacles(self, obstacles):
        """Publish obstacle markers for visualization"""
        marker_array = MarkerArray()
        
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "limo_base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = obstacle['center'][0]
            marker.pose.position.y = obstacle['center'][1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = obstacle['radius'] * 2
            marker.scale.y = obstacle['radius'] * 2
            marker.scale.z = 0.5
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker_array.markers.append(marker)
            
        self.obstacle_pub.publish(marker_array)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ObstacleDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass