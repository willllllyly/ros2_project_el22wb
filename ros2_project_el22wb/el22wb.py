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
        self.patrol_points = [
            #(0.413, -1.53), (-2.41, -4.06),
            (-5.31, -2.32, 2.9), #(0.361, -5.93),  
            (2.73, -6.45, 0.0), #(2.15, -10.4), 
            (-1.97, -9.93, 2.9)]
        self.current_patrol_index = 0

        # Initialise
        self.bridge = CvBridge()
        self.pre_callback()
        
        # Empty array to store images in
        self.video_array = []


        # Create a subscriber to the camera/image_raw topic
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.callback, 10)
        self.subscription # Prevent unused variable warning 
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.navigate_patrol()

    def pre_callback(self):

        # Initialise HSV values
        self.hsv_red0_lower = np.array([0 - self.sensitivity, 100, 100]) 
        self.hsv_red0_upper = np.array([0 + self.sensitivity, 255, 255])
        #self.hsv_red180_lower = np.array([180 - (self.sensitivity/2), 100, 100]) 
        #self.hsv_red180_upper = np.array([180, 255, 255])
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        self.hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        self.hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
    
        # Create new window and define the size
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('camera_Feed', 320, 240)        
        
        
    def callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            # Convert the image to HSV colour space
            self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            
            # Create masks to filter different colours
            #self.red0_mask = cv2.inRange(self.hsv_image, self.hsv_red0_lower, self.hsv_red180_upper)
            #self.red180_mask = cv2.inRange(self.hsv_image, self.hsv_red0_upper, self.hsv_red180_lower)
            self.red_mask = cv2.inRange(self.hsv_image, self.hsv_red0_lower, self.hsv_red0_upper)
            self.green_mask = cv2.inRange(self.hsv_image, self.hsv_green_lower, self.hsv_green_upper)
            self.blue_mask = cv2.inRange(self.hsv_image, self.hsv_blue_lower, self.hsv_blue_upper)
            
            # Combine masks
            #self.red_mask = cv2.bitwise_or(self.red0_mask, self.red180_mask)
            
            # Apply the masks to original image
           # filtered_img = cv2.bitwise_and(self.cv_image, self.cv_image, mask=self.red_mask | self.green_mask | self.blue_mask)  
            

            # Find contours in the mask
            contours_r, _ = cv2.findContours(self.red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_g, _ = cv2.findContours(self.green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_b, _ = cv2.findContours(self.blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(len(contours_r))
            print(len(contours_g))
            print(len(contours_b))
            
            if len(contours_r) > 0:
                # Find the largest contour
                largest_contour_r = max(contours_r, key=cv2.contourArea)
                
                self.red_found = True
                    
                # Get the enclosing circle for the largest contour
                (x1, y1), radius_r = cv2.minEnclosingCircle(largest_contour_r)
                center_x1, center_y1 = int(x1), int(y1)
                
                # Draw the circle around detected object
                cv2.circle(self.cv_image, (center_x1, center_y1), int(radius_r), (0, 255, 0), 2)
                    
                # Print a message
                print("Red object detected!")    
                
                # Compute the area of the largest contour
                #contour_r_area = cv2.contourArea(largest_contour_r)
                
                # Set a threshold for object detection
                #if contour_r_area > 500:  # Adjust the threshold based on testing
                    
                   
            else:
                self.red_found = False

            if len(contours_g) > 0:
                # Find the largest contour
                largest_contour_g = max(contours_g, key=cv2.contourArea)
                
                self.green_found = True
                    
                # Get the enclosing circle for the largest contour
                (x2, y2), radius_g = cv2.minEnclosingCircle(largest_contour_g)
                center_x2, center_y2 = int(x2), int(y2)
                    
                # Draw the circle around detected object
                cv2.circle(self.cv_image, (center_x2, center_y2), int(radius_g), (0, 255, 0), 2)
                    
                # Print a message
                print("Green object detected!")                
                # Compute the area of the largest contour
                #contour_g_area = cv2.contourArea(largest_contour_g)
                
                # Set a threshold for object detection
                #if contour_g_area > 500:  # Adjust the threshold based on testing
 
                    
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
                    cv2.circle(self.cv_image, (center_x3, center_y3), int(radius_b), (0, 255, 0), 2)
                    
                    # Print a message
                    print("Blue object detected!")
                    
            else:
                self.blue_found = False
            
            # Display the result
            cv2.imshow('camera_Feed', self.cv_image)
            cv2.waitKey(3)
                
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        #self.get_logger().info(f"Navigation Feedback: {feedback}")
    
    def send_goal(self, x, y, orienation):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(orienation / 2)
        goal_msg.pose.pose.orientation.w = cos(orienation / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)
    
    def navigate_patrol(self):
        if self.current_patrol_index < len(self.patrol_points):
            x, y, orientation = self.patrol_points[self.current_patrol_index]
            self.get_logger().info(f"Navigating to patrol point: {x}, {y}")
            self.send_goal(x, y, orientation)
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
        
        # Move to the next patrol point if available
        if self.current_patrol_index < len(self.patrol_points):
            x, y, orientation = self.patrol_points[self.current_patrol_index]
            self.get_logger().info(f"Moving to next patrol point: {x}, {y}")
            self.current_patrol_index += 1
            self.send_goal(x, y, orientation)  # Send next goal
        else:
            self.get_logger().info("All patrol points visited. Stopping.")
              
        


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
