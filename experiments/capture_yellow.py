#!/usr/bin/env python3  

import os  
import cv2  
import numpy as np  
from picamera2 import Picamera2  
import time  

class YellowBallDetector:  
    def __init__(self):  
        # HSV color ranges for yellow detection  
        # Adjust these parameters based on your specific lighting and ball  
        self.lower_yellow = np.array([20, 100, 100])   # Lower HSV threshold  
        self.upper_yellow = np.array([30, 255, 255])   # Upper HSV threshold  

    def detect_yellow_ball(self, frame):  
        """  
        Advanced yellow ball detection method  
        """  
        # Convert to HSV color space  
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  
        
        # Create yellow color mask  
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)  
        
        # Optional: Apply morphological operations to reduce noise  
        kernel = np.ones((5,5), np.uint8)  
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)  
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)  
        
        # Find contours  
        contours, _ = cv2.findContours(  
            yellow_mask,   
            cv2.RETR_EXTERNAL,   
            cv2.CHAIN_APPROX_SIMPLE  
        )  
        
        # Detected ball regions  
        detected_balls = []  
        
        # Process each contour  
        for contour in contours:  
            # Calculate contour area  
            area = cv2.contourArea(contour)  
            
            # Filter contours by area to reduce noise  
            if 100 < area < 5000:  # Adjust these values based on your ball size  
                # Get bounding rectangle  
                x, y, w, h = cv2.boundingRect(contour)  
                
                # Optional: Check aspect ratio for circular shape  
                aspect_ratio = w / float(h)  
                if 0.8 < aspect_ratio < 1.2:  
                    detected_balls.append((x, y, w, h))  
        
        return detected_balls  

def main():  
    # Initialize Picamera2  
    picam2 = Picamera2()  
    config = picam2.create_preview_configuration(  
        main={"size": (1280, 720)},  
        lores={"size": (640, 480)}  
    )  
    picam2.configure(config)  
    picam2.start()  
    
    # Create detector  
    detector = YellowBallDetector()  
    
    try:  
        while True:  
            # Capture frame  
            frame = picam2.capture_array()  
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

            # Make a copy for drawing  
            display_frame = frame.copy()  
            
            # Detect yellow balls  
            balls = detector.detect_yellow_ball(frame)  
            
            # Draw rectangles  
            for (x, y, w, h) in balls:  
                cv2.rectangle(  
                    display_frame,   
                    (x, y),   
                    (x+w, y+h),   
                    (0, 0, 255),  # Red color  
                    2  # Thickness  
                )  
            
            # Visualize the mask for debugging  
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  
            yellow_mask = cv2.inRange(  
                hsv,   
                detector.lower_yellow,   
                detector.upper_yellow  
            )  
            
            # Display frames  
            cv2.imshow('Yellow Ball Detection', display_frame)  
            cv2.imshow('Yellow Color Mask', yellow_mask)  
            
            # Wait for key press  
            key = cv2.waitKey(1) & 0xFF  
            if key == ord('q'):  
                break  
    
    except Exception as e:  
        print(f"Error: {e}")  
    
    finally:  
        # Cleanup  
        picam2.stop()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()