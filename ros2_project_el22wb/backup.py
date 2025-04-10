# Exercise 1 - Display an image of the camera feed to the screen

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from math import sin, cos
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class TurtleBotNavigator(Node):
    def __init__(self):
        super().__init__('cI')
        
        # Initialise flags to signal colour detection
        self.red_found = False
        self.green_found = False
        self.blue_found = False

        # Sensitivity value for colour detection
        self.sensitivity = 10

        # Set the co-ordinates for motion planning
        self.patrol_points = [(-5.31, -2.32), (2.73, -6.45), (-0.69, -10)]
        self.current_patrol_index = 0

        # Initialise
        self.bridge = CvBridge()

        # Create a subscriber to the camera/image_raw topic
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.callback, 10)
        self.subscription # Prevent unused variable warning 
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.navigate_patrol()
        
    def callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            # Convert the image to HSV colour space
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Initialise HSV values
            hsv_red0_lower = np.array([0 + (self.sensitivity/2), 100, 100]) 
            hsv_red0_upper = np.array([0 + (self.sensitivity/2), 255, 255])
            hsv_red180_lower = np.array([180 - (self.sensitivity/2), 100, 100]) 
            hsv_red180_upper = np.array([180 - (self.sensitivity/2), 255, 255])
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
            hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
            hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

            # Create masks to filter different colours
            red0_mask = cv2.inRange(hsv_image, hsv_red0_lower, hsv_red0_upper)
            red180_mask = cv2.inRange(hsv_image, hsv_red180_lower, hsv_red180_upper)
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

            # Combine masks
            red_mask = cv2.bitwise_or(red0_mask, red180_mask)

            # Display the image
            filtered_img = cv2.bitwise_and(hsv_image, hsv_image, mask=None)
            cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL) 
            cv2.imshow('camera_Feed', filtered_img)
            cv2.resizeWindow('camera_Feed', 320, 240) 
            cv2.waitKey(3)

            # Find contours in the mask
            contours_r, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_g, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_b, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours_r) > 0:
                # Find the largest contour
                largest_contour_r = max(contours_r, key=cv2.contourArea)
                
                # Compute the area of the largest contour
                contour_r_area = cv2.contourArea(largest_contour_r)
                
                # Set a threshold for object detection
                if contour_r_area > 500:  # Adjust the threshold based on testing
                    self.red_found = True
                    
                    # Get the enclosing circle for the largest contour
                    (x1, y1), radius_r = cv2.minEnclosingCircle(largest_contour_r)
                    center_x1, center_y1 = int(x1), int(y1)
                    
                    # Draw the circle around detected object
                    cv2.circle(filtered_img, (center_x1, center_y1), int(radius_r), (0, 255, 0), 2)
                    
                    # Print a message
                    print("Red object detected!")
                    
            else:
                self.red_found = False

            if len(contours_g) > 0:
                # Find the largest contour
                largest_contour_g = max(contours_g, key=cv2.contourArea)
                
                # Compute the area of the largest contour
                contour_g_area = cv2.contourArea(largest_contour_g)
                
                # Set a threshold for object detection
                if contour_g_area > 500:  # Adjust the threshold based on testing
                    self.green_found = True
                    
                    # Get the enclosing circle for the largest contour
                    (x2, y2), radius_g = cv2.minEnclosingCircle(largest_contour_g)
                    center_x2, center_y2 = int(x2), int(y2)
                    
                    # Draw the circle around detected object
                    cv2.circle(filtered_img, (center_x2, center_y2), int(radius_g), (0, 255, 0), 2)
                    
                    # Print a message
                    print("Green object detected!")
                    
            else:
                self.green_found = False
            
            if len(contours_b) > 0:
                # Find the largest contour
                largest_contour_b = max(contours_b, key=cv2.contourArea)
                
                # Compute the area of the largest contour
                contour_b_area = cv2.contourArea(largest_contour_b)
                
                # Set a threshold for object detection
                if contour_b_area > 500:  # Adjust the threshold based on testing
                    self.blue_found = True
                    
                    # Get the enclosing circle for the largest contour
                    (x3, y3), radius_b = cv2.minEnclosingCircle(largest_contour_b)
                    center_x3, center_y3 = int(x3), int(y3)
                    
                    # Draw the circle around detected object
                    cv2.circle(filtered_img, (center_x3, center_y3), int(radius_b), (0, 255, 0), 2)
                    
                    # Print a message
                    print("Green object detected!")
                    
            else:
                self.blue_found = False
            
            # Display the result
            cv2.imshow('Processed Image', filtered_img)
            cv2.waitKey(3) 
                
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Navigation Feedback: {feedback}")
    
    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)
    
    def navigate_patrol(self):
        if self.current_patrol_index < len(self.patrol_points):
            x, y = self.patrol_points[self.current_patrol_index]
            self.get_logger().info(f"Navigating to patrol point: {x}, {y}")
            self.send_goal(x, y, 0.0)
            self.current_patrol_index += 1

    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        


def main():

    def signal_handler(sig, frame):
        rclpy.shutdown()

    rclpy.init(args=None)
    cI = TurtleBotNavigator()


    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Destroy all image windows before closing node
    cv2.destroyAllWindows()
    

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
