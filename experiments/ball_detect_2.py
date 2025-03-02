import cv2  
import numpy as np  
from picamera2 import Picamera2  
import logging  
import time  

class YellowBallDetector:  
    def __init__(self,   
                 weights_path='yolov4-tiny.weights',   
                 config_path='yolov4-tiny.cfg',   
                 coco_path='coco.names'):  
        # Logging setup  
        logging.basicConfig(level=logging.INFO,   
                            format='%(asctime)s - %(levelname)s: %(message)s')  
        self.logger = logging.getLogger(__name__)  

        # Yellow Ball Specific Color Range (HSV)  
        # self.YELLOW_LOWER = np.array([20, 100, 100])  
        # self.YELLOW_UPPER = np.array([35, 255, 255])  

        self.YELLOW_LOWER = np.array([15, 50, 50])   # More lenient lower bound  
        self.YELLOW_UPPER = np.array([40, 255, 255]) # More lenient upper bound  

        # Load class names  
        try:  
            with open(coco_path, 'r') as f:  
                self.classes = [line.strip() for line in f.readlines()]  
        except FileNotFoundError:  
            self.logger.error(f"Class names file not found: {coco_path}")  
            self.classes = []  

    def detect_yellow_ball(self, frame):  
        """  
        Specialized yellow ball detection method using color  
        """  
        # Convert to HSV color space  
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        
        # Create yellow color mask  
        yellow_mask = cv2.inRange(hsv_frame, self.YELLOW_LOWER, self.YELLOW_UPPER)  
        
        # Visualize the mask (optional)  
        cv2.imshow('Yellow Mask', yellow_mask)  
        
        # Find contours of yellow objects  
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        
        # Filter and process contours  
        yellow_balls = []  
        for contour in contours:  
            # Filter contours by area to eliminate noise  
            area = cv2.contourArea(contour)  
            if area > 90:  # Adjust this threshold as needed  
                x, y, w, h = cv2.boundingRect(contour)  
                
                # Optional: Add aspect ratio check to filter round objects  
                aspect_ratio = w / float(h)  
                if 0.8 <= aspect_ratio <= 1.2:  
                    yellow_balls.append((x, y, w, h))  
                    self.logger.info(f"Color Detection: Found yellow ball - Area: {area}, Aspect Ratio: {aspect_ratio}")  
        
        return yellow_balls  

    
    def process_frame(self, frame):  
        """  
        Combine color-based and YOLO detection methods  
        """  
        # Color-based yellow ball detection  
        color_detections = self.detect_yellow_ball(frame)  
        self.logger.info(f"Color Detections: {len(color_detections)}")  
        
        # Draw color-based detections in RED  
        for (x, y, w, h) in color_detections:  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  
            cv2.putText(frame, "Yellow Ball (Color)", (x, y-10),   
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
        
        return frame  

def main():  
    # Camera setup  
    picam2 = Picamera2()  
    config = picam2.create_video_configuration(  
        main={"size": (416, 416), "format": "RGB888"}  
    )  
    picam2.configure(config)  
    picam2.start()  

    # Initialize detector  
    ball_detector = YellowBallDetector()  

    try:  
        while True:  
            frame = picam2.capture_array()  
            
            if frame is not None:  
                processed_frame = ball_detector.process_frame(frame)  
                cv2.imshow('Yellow Ball Detection', processed_frame)  
                cv2.imshow('Detection Mask', cv2.cvtColor(  
                    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),   
                    cv2.COLOR_HSV2BGR  
                ))  
            
            # Break on 'q' key  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  

    except Exception as e:  
        print(f"An error occurred: {e}")  
    
    finally:  
        picam2.stop()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()