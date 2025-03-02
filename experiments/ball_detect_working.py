import cv2  
import numpy as np  
from picamera2 import Picamera2  
import logging  

class BallDetector:  
    def __init__(self,   
                 weights_path='yolov4-tiny.weights',   
                 config_path='yolov4-tiny.cfg',   
                 coco_path='coco.names'):  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  

        # Color Detection Parameters  
        self.YELLOW_LOWER = np.array([20, 100, 100])  
        self.YELLOW_UPPER = np.array([35, 255, 255])  

        # YOLO Setup  
        try:  
            self.net = cv2.dnn.readNet(weights_path, config_path)  
            
            # Layer names handling  
            layer_names = self.net.getLayerNames()  
            self.output_layers = [  
                layer_names[i - 1] for i in   
                self.net.getUnconnectedOutLayers().flatten()  
            ]  

            # Load class names  
            with open(coco_path, 'r') as f:  
                self.classes = [line.strip() for line in f.readlines()]  
        except Exception as e:  
            self.logger.error(f"YOLO initialization error: {e}")  
            self.net = None  
            self.classes = []  

    def color_detection(self, frame):  
        """Pure color-based yellow ball detection"""  
        # Convert to HSV  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        
        # Create mask  
        mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)  
        
        # Find contours  
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        
        # Filter contours  
        color_detections = []  
        for contour in contours:  
            area = cv2.contourArea(contour)  
            if area > 50:  # Minimum area threshold  
                x, y, w, h = cv2.boundingRect(contour)  
                color_detections.append((x, y, w, h))  
        
        return color_detections, mask  

    def yolo_detection(self, frame):  
        """YOLO-based object detection"""  
        if self.net is None:  
            return []  

        height, width, _ = frame.shape  
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  
        
        self.net.setInput(blob)  
        outs = self.net.forward(self.output_layers)  
        
        ball_detections = []  
        for out in outs:  
            for detection in out:  
                scores = detection[5:]  
                class_id = np.argmax(scores)  
                confidence = scores[class_id]  
                
                # Filter for ball-like objects  
                if confidence > 0.1 and self.classes[class_id] in ['sports ball', 'ball']:  
                    center_x = int(detection[0] * width)  
                    center_y = int(detection[1] * height)  
                    w = int(detection[2] * width)  
                    h = int(detection[3] * height)  
                    
                    x = int(center_x - w / 2)  
                    y = int(center_y - h / 2)  
                    
                    ball_detections.append((x, y, w, h))  
        
        return ball_detections  

    def process_frame(self, frame):  
        # Color Detection  
        color_detections, mask = self.color_detection(frame)  
        
        # YOLO Detection  
        yolo_detections = self.yolo_detection(frame)  
        
        # Draw Color Detections (RED)  
        # for (x, y, w, h) in color_detections:  
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  
        #     cv2.putText(frame, "Yellow Ball (Color)", (x, y-10),   
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
        
        # Draw YOLO Detections (GREEN)  
        for (x, y, w, h) in yolo_detections:  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
            cv2.putText(frame, "Ball (YOLO)", (x, y-10),   
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
        
        # Display mask for debugging  
        cv2.imshow('Color Mask', mask)  
        
        return frame  

def main():  
    picam2 = Picamera2()  
    config = picam2.create_video_configuration(  
        main={"size": (416, 416), "format": "RGB888"}  
    )  
    picam2.configure(config)  
    picam2.start()  

    detector = BallDetector()  

    try:  
        while True:  
            frame = picam2.capture_array()  
            
            if frame is not None:  
                processed_frame = detector.process_frame(frame)  
                cv2.imshow('Ball Detection', processed_frame)  
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  

    except Exception as e:  
        print(f"Error: {e}")  
    
    finally:  
        picam2.stop()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()