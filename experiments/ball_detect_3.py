import cv2  
import numpy as np  
from picamera2 import Picamera2  

class YellowBallDetector:  
    def __init__(self):  
        self.YELLOW_LOWER = np.array([20, 100, 100])  
        self.YELLOW_UPPER = np.array([30, 255, 255])  

    def detect_yellow_ball(self, frame):  
        # Convert to HSV color space  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        
        # Apply Gaussian blur to reduce noise  
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)  
        
        # Create mask for yellow color  
        yellow_mask = cv2.inRange(blurred, self.YELLOW_LOWER, self.YELLOW_UPPER)  
        
        # Morphological operations to remove noise  
        kernel = np.ones((5,5), np.uint8)  
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)  
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)  
        
        # Find contours  
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        
        # Initialize variables for best ball detection  
        best_ball = None  
        max_area = 0  
        
        # Process contours  
        for contour in contours:  
            # Calculate area and skip small contours  
            area = cv2.contourArea(contour)  
            if area < 90:  
                continue  
            
            # Calculate bounding rectangle and aspect ratio  
            x, y, w, h = cv2.boundingRect(contour)  
            aspect_ratio = float(w) / h  
            
            # Check if contour matches ball-like shape  
            if 0.8 <= aspect_ratio <= 1.2:  
                # Keep track of the largest valid contour  
                if area > max_area:  
                    max_area = area  
                    best_ball = contour  
        
        # Return the best detected ball contour  
        return best_ball, yellow_mask  

    def process_frame(self, frame):  
        # Detect yellow ball  
        ball_contour, yellow_mask = self.detect_yellow_ball(frame)  
        
        # If a ball is detected, draw its contour  
        if ball_contour is not None:  
            cv2.drawContours(frame, [ball_contour], -1, (0, 255, 0), 2)  
        
        return frame, yellow_mask  

# Example usage  
def main():  
    # Initialize camera or video capture  
    picam2 = Picamera2()  
    config = picam2.create_video_configuration(  
        main={"size": (416, 416), "format": "RGB888"}  
    )  
    picam2.configure(config)  
    picam2.start()  

    
    detector = YellowBallDetector()  
    
    while True:  
        # Read frame from camera  
        frame = picam2.capture_array()  
       
        # Process the frame  
        processed_frame, yellow_mask = detector.process_frame(frame)  
        
        # Display results  
        cv2.imshow('Original', processed_frame)  
        cv2.imshow('Yellow Mask', yellow_mask)  
        
        # Exit on 'q' key press  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
    
    # Release resources  
    cap.release()  
    cv2.destroyAllWindows()  

if __name__ == '__main__':  
    main()  