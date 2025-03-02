#!/usr/bin/env python3  

import os  
import cv2  
import time  
import uuid  
import numpy as np  
from picamera2 import Picamera2  

class YellowBallDetector:  
    def __init__(self,   
                 dataset_dir='/home/usman/yellow_ball_dataset',   
                 max_images=50):  
        """  
        Initialize Yellow Ball Detector  
        """  
        # Camera setup  
        self.picam2 = Picamera2()  
        config = self.picam2.create_preview_configuration(  
            main={"size": (1280, 720)}  
        )  
        self.picam2.configure(config)  

        # Dataset directory  
        self.dataset_dir = dataset_dir  
        os.makedirs(dataset_dir, exist_ok=True)  

        # Optimized HSV Parameters  
        self.lower_yellow = np.array([19, 0, 161])   # LH, LS, LV  
        self.upper_yellow = np.array([34, 255, 255])  # UH, US, UV  

        # Capture tracking  
        self.max_images = max_images  
        self.image_count = 0  

    def detect_yellow_ball(self, frame):  
        """  
        Detect yellow ball with advanced filtering  
        
        Args:  
            frame (numpy.ndarray): Input image frame  
        
        Returns:  
            tuple: Detected ball information and colored mask  
        """  
        # Convert to HSV  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        
        # Create yellow mask  
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)  
        
        # Morphological operations to reduce noise  
        kernel = np.ones((5,5), np.uint8)  
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)  
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)  
        
        # Find contours  
        contours, _ = cv2.findContours(  
            yellow_mask,   
            cv2.RETR_EXTERNAL,   
            cv2.CHAIN_APPROX_SIMPLE  
        )  
        
        # Best ball candidate  
        best_ball = None  
        max_white_percentage = 0  
        
        # Process each contour  
        for contour in contours:  
            # Calculate contour area  
            area = cv2.contourArea(contour)  
            
            # Filter contours by area to reduce noise  
            if 500 < area < 10000:  # Adjust based on expected ball size  
                # Get bounding rectangle  
                x, y, w, h = cv2.boundingRect(contour)  
                
                # Create a mask for this specific contour  
                contour_mask = np.zeros_like(yellow_mask)  
                cv2.drawContours(  
                    contour_mask,   
                    [contour],   
                    -1,   
                    255,   
                    -1  
                )  
                
                # Calculate white percentage  
                contour_region = cv2.bitwise_and(  
                    yellow_mask,   
                    yellow_mask,   
                    mask=contour_mask  
                )  
                white_pixels = cv2.countNonZero(contour_region)  
                total_pixels = cv2.countNonZero(contour_mask)  
                white_percentage = white_pixels / total_pixels * 100  
                
                # Optional: Check aspect ratio for circular shape  
                aspect_ratio = w / float(h)  
                
                # Criteria for best ball  
                if (  
                    0.8 < aspect_ratio < 1.2 and  # Nearly circular  
                    white_percentage > max_white_percentage and  
                    white_percentage > 70  # At least 70% white  
                ):  
                    max_white_percentage = white_percentage  
                    best_ball = (x, y, w, h, white_percentage)  
        
        # Create a 3-channel mask for drawing  
        mask_colored = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)  
        
        # Draw rectangles directly on the mask if ball detected  
        if best_ball:  
            x, y, w, h, _ = best_ball  
            cv2.rectangle(  
                mask_colored,   
                (x, y),   
                (x+w, y+h),   
                (0, 255, 0),  # Green color  
                2  
            )  
        
        return best_ball, mask_colored  

    def capture_annotated_dataset(self):  
        """  
        Interactive annotated dataset capture  
        """  
        # Start camera  
        self.picam2.start()  
        time.sleep(2)  # Warm-up time  
        
        try:  
            while self.image_count < self.max_images:  
                # Capture frame  
                frame = self.picam2.capture_array()  
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                
                # Create display frame  
                display_frame = frame.copy()  
                
                # Detect ball and get mask  
                ball_info, colored_mask = self.detect_yellow_ball(frame)  
                
                # Prepare display information  
                status_text = f"Images: {self.image_count}/{self.max_images}"  
                instruction_text = "Press 'c' to capture, 'q' to quit"  
                
                # Draw ball if detected  
                if ball_info:  
                    x, y, w, h, white_percentage = ball_info  
                    
                    # Draw GREEN rectangle on original frame  
                    cv2.rectangle(  
                        display_frame,   
                        (x, y),   
                        (x+w, y+h),   
                        (0, 255, 0),  # Green color  
                        2  
                    )  
                    
                    # Add white percentage text  
                    cv2.putText(  
                        display_frame,   
                        f"White: {white_percentage:.2f}%",   
                        (x, y-10),   
                        cv2.FONT_HERSHEY_SIMPLEX,   
                        0.9,   
                        (0, 255, 0),   
                        2  
                    )  
                
                # Add status and instruction text  
                cv2.putText(  
                    display_frame,   
                    status_text,   
                    (10, 30),   
                    cv2.FONT_HERSHEY_SIMPLEX,   
                    1,   
                    (0, 255, 0),   
                    2  
                )  
                cv2.putText(  
                    display_frame,   
                    instruction_text,   
                    (10, 70),   
                    cv2.FONT_HERSHEY_SIMPLEX,   
                    1,   
                    (0, 255, 0),   
                    2  
                )  
                
                # Display frames  
                cv2.namedWindow('Ball Annotation', cv2.WINDOW_NORMAL)  
                cv2.resizeWindow('Ball Annotation', 1280, 720)  
                cv2.imshow('Ball Annotation', display_frame)  
                
                # HSV Mask Display with Rectangles  
                cv2.namedWindow('HSV Mask', cv2.WINDOW_NORMAL)  
                cv2.resizeWindow('HSV Mask', 1280, 720)  
                cv2.imshow('HSV Mask', colored_mask)  
                
                # Wait for key press  
                key = cv2.waitKey(1) & 0xFF  
                key = input ("Enter key - c to continue, q to exit")
                # Capture image on 'c' key  
                if key == ord('c'):  
                    if ball_info:  
                        # Generate unique filename  
                        filename = os.path.join(  
                            self.dataset_dir,   
                            f'yellow_ball_{uuid.uuid4()}.jpg'  
                        )  
                        
                        # Annotate frame with bounding boxes before saving  
                        x, y, w, h, white_percentage = ball_info  
                        
                        # GREEN rectangle for detection  
                        cv2.rectangle(  
                            frame,   
                            (x, y),   
                            (x+w, y+h),   
                            (0, 255, 0),  # Green color  
                            2  
                        )  
                        
                        # Add text  
                        cv2.putText(  
                            frame,   
                            f"White: {white_percentage:.2f}%",   
                            (x, y-10),   
                            cv2.FONT_HERSHEY_SIMPLEX,   
                            0.9,   
                            (0, 255, 0),   
                            2  
                        )  
                        
                        # Save annotated image  
                        cv2.imwrite(filename, frame)  
                        
                        # Increment counter  
                        self.image_count += 1  
                        print(f"Saved image: {filename}")  
                        
                        # Brief pause to prevent multiple captures  
                        time.sleep(0.5)  
                    else:  
                        print("No yellow ball detected. Reposition and try again.")  
                
                # Quit on 'q' key  
                elif key == ord('q'):  
                    break  
        
        except Exception as e:  
            print(f"Capture error: {e}")  
        
        finally:  
            # Cleanup  
            self.picam2.stop()  
            cv2.destroyAllWindows()  
        
        print(f"Annotation complete. Total images: {self.image_count}")  

def main():  
    # Create detector instance  
    detector = YellowBallDetector()  
    
    # Start interactive detection  
    detector.capture_annotated_dataset()  

if __name__ == "__main__":  
    main()