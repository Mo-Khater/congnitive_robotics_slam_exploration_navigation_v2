#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge

def extract_features(img):
    img_norm = img.astype(np.float32) / 255.0
    color_hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    color_hist = cv2.normalize(color_hist, color_hist).flatten()
    mean_color = np.mean(img_norm, axis=(0,1))
    std_color = np.std(img_norm, axis=(0,1))
    return np.concatenate([color_hist, mean_color, std_color])

def cosine(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den if den != 0 else 0

class Detector:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.bridge = CvBridge()
        self.goal_found = False  # Track if we already found it
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_path = os.path.join(script_dir, '..', 'images', 'blue_ball.jpg')
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            ref_img = cv2.resize(ref_img, (100,100))
            self.ref_features = extract_features(ref_img)
            rospy.loginfo(f"{robot_name}: 🔍 Detector initialized - searching for goal...")
        else:
            self.ref_features = None
            rospy.logerr(f"{robot_name}: ❌ Failed to load reference image!")
            
        rospy.Subscriber(f"/{robot_name}/color/image_raw", Image, self.cb)
        self.pub = rospy.Publisher(f"/{robot_name}/goal_detected", Bool, queue_size=1)
        self.sim_pub = rospy.Publisher(f"/{robot_name}/goal_similarity", Float32, queue_size=1)
        self.th = 0.5

    def print_victory_message(self, similarity):
        """Print an epic victory message!"""
        rospy.logwarn("\n" + "="*80)
        rospy.logwarn("██████╗  ██████╗  █████╗ ██╗         ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ")
        rospy.logwarn("██╔════╝ ██╔═══██╗██╔══██╗██║         ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗")
        rospy.logwarn("██║  ███╗██║   ██║███████║██║         █████╗  ██║   ██║██║   ██║█████╗  ██║  ██║")
        rospy.logwarn("██║   ██║██║   ██║██╔══██║██║         ██╔══╝  ██║   ██║██║   ██║██╔══╝  ██║  ██║")
        rospy.logwarn("╚██████╔╝╚██████╔╝██║  ██║███████╗    ██║     ╚██████╔╝╚██████╔╝███████╗██████╔╝")
        rospy.logwarn(" ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝      ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ")
        rospy.logwarn("="*80)
        rospy.logwarn(f"🎉🎉🎉 {self.robot_name.upper()} SUCCESSFULLY FOUND THE GOAL! 🎉🎉🎉")
        rospy.logwarn(f"✨ Confidence Score: {similarity:.3f}")
        rospy.logwarn(f"🤖 Robot: {self.robot_name}")
        rospy.logwarn(f"⏰ Time: {rospy.Time.now().to_sec():.2f}s since start")
        rospy.logwarn("🏆 MISSION ACCOMPLISHED! 🏆")
        rospy.logwarn("="*80 + "\n")

    def cb(self, msg):
        if self.ref_features is None:
            return
            
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([90, 50, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        sims=[]
        for c in cnts[:10]:
            if cv2.contourArea(c) < 100:
                continue
            x,y,w,h = cv2.boundingRect(c)
            if h < 10 or w < 10:
                continue
            obj = img[y:y+h, x:x+w]
            obj = cv2.resize(obj, (100,100))
            obj_features = extract_features(obj)
            sim = cosine(obj_features, self.ref_features)
            sims.append(sim)

        m = max(sims) if sims else 0
        scaled = min(1.0, m * 10000)
        self.sim_pub.publish(float(scaled))
        
        detected = scaled > self.th
        self.pub.publish(detected)
        
        # Print victory message once when first detected
        if detected and not self.goal_found:
            self.goal_found = True
            self.print_victory_message(scaled)

def main():
    rospy.init_node("det", anonymous=True)
    robot_name = rospy.get_param('~robot_name', 'robot1')
    Detector(robot_name)
    rospy.spin()

if __name__ == "__main__":
    main()