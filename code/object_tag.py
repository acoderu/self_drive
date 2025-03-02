#!/usr/bin/env python3  

# Standard library imports  
import os  
import sys  
import time  
import signal  
import logging  
import argparse  
import traceback  

# Third-party library imports  
import cv2  
import numpy as np  
import libcamera  
from picamera2 import Picamera2  
import tflite_runtime.interpreter as tflite  
from PIL import Image  

# Configuration Constants  
BASE_DIR = "/home/usman/tracking"  
IM_WIDTH = 640  
IM_HEIGHT = 480  

# Visualization Constants  
font = cv2.FONT_HERSHEY_SIMPLEX  
bottomLeftCornerOfText = (10, IM_HEIGHT-10)  
fontScale = 1  
fontColor = (255, 255, 255)  # white  
boxColor = (0, 0, 255)       # RED  
boxLineWidth = 2  
lineType = 2  

# Global variable to control main loop  
running = True  

def signal_handler(signum, frame):  
    """Handle interrupt signals (Ctrl+C)"""  
    global running  
    print("\nInterrupt received, shutting down...")  
    running = False  

# Register signal handlers  
signal.signal(signal.SIGINT, signal_handler)    # Ctrl+C  
signal.signal(signal.SIGTERM, signal_handler)   # Termination signal  

# Configure logging  
logging.basicConfig(  
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s: %(message)s',  
    handlers=[  
        logging.FileHandler(os.path.join(BASE_DIR, 'object_detection.log'), mode='w'),  
        logging.StreamHandler(sys.stdout)  
    ]  
)  
logger = logging.getLogger(__name__)  

class RaspberryPiObjectDetection:  
    def __init__(self, model_path, model_labels, confidence_threshold=0.6):  
        """Initialize the object detection system for Raspberry Pi"""  
        logger.info("===== INITIALIZING OBJECT DETECTION =====")  
        
        try:  
            # Camera configuration  
            self.width = IM_WIDTH  
            self.height = IM_HEIGHT  
            self.confidence_threshold = confidence_threshold  
            
            # Validate model path  
            if not os.path.exists(model_path):  
                logger.error(f"Model file not found: {model_path}")  
                raise FileNotFoundError(f"Model file not found: {model_path}")  
            
            # Initialize TFLite Interpreter  
            self.interpreter = tflite.Interpreter(model_path=model_path)  
            
            # Allocate tensors  
            self.interpreter.allocate_tensors()  

            # Get input and output details  
            self.input_details = self.interpreter.get_input_details()  
            self.output_details = self.interpreter.get_output_details()  

            # Log detailed input and output tensor information  
            self._log_tensor_details()  

            # Initialize Picamera2  
            self.picam2 = Picamera2()  
            
            # Configure camera with advanced settings  
            config = self.picam2.create_still_configuration(  
                main={"size": (self.width, self.height)},  
                lores={"size": (640, 480)}  
            )  
            self.picam2.configure(config)  

            # Load class names  
            self.class_names = self.load_class_names(os.path.join(BASE_DIR, model_labels))  
            
            logger.info("Initialization complete")  
        
        except Exception as e:  
            logger.error(f"Initialization failed: {e}")  
            logger.error(traceback.format_exc())  
            raise  

    def _log_tensor_details(self):  
        """Log detailed information about input and output tensors"""  
        logger.info("Input Tensor Details:")  
        for i, detail in enumerate(self.input_details):  
            logger.info(f"Input {i}:")  
            logger.info(f"  Shape: {detail['shape']}")  
            logger.info(f"  dtype: {detail['dtype']}")  
            logger.info(f"  Quantization: {detail.get('quantization', 'N/A')}")  
        
        logger.info("Output Tensor Details:")  
        for i, detail in enumerate(self.output_details):  
            logger.info(f"Output {i}:")  
            logger.info(f"  Shape: {detail['shape']}")  
            logger.info(f"  dtype: {detail['dtype']}")  

    def load_class_names(self, filename):  
        """Load class names from a file"""  
        try:  
            with open(filename, 'r') as f:  
                return {i: line.strip() for i, line in enumerate(f.readlines())}  
        except Exception as e:  
            logger.error(f"Failed to load class names: {e}")  
            return {}  

    def capture_image(self):  
        """Capture image using Picamera2 with robust error handling"""  
        logger.info("===== STARTING IMAGE CAPTURE PROCESS =====")  
        
        try:  
            # Start the camera  
            self.picam2.start()  
            
            # Allow camera to stabilize  
            time.sleep(1)  
            
            # Capture image  
            logger.info("Capturing image...")  
            image = self.picam2.capture_array()  
            
            # Stop the camera  
            self.picam2.stop()  
            
            if image is None:  
                logger.error("Failed to capture image: Returned None")  
                return None  
            
            # Convert from BGR to RGB   
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            
            logger.info(f"Image captured. Shape: {image.shape}")  
            return image  
        
        except Exception as e:  
            logger.error(f"Image capture failed: {e}")  
            logger.error(traceback.format_exc())  
            return None  

    def preprocess_image(self, image):  
        """Preprocess image for TFLite model (Quantized MobileNet SSD)"""  
        # Resize to model's input size (typically 300x300 for MobileNet SSD)  
        input_shape = self.input_details[0]['shape']  
        resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))  
        
        # Convert to uint8 WITHOUT normalization  
        input_data = resized_image.astype(np.uint8)  
        
        # Add batch dimension  
        input_data = np.expand_dims(input_data, axis=0)  
        
        return input_data  

    def detect_objects(self, image):  
        """Detect objects in the image with TFLite"""  
        try:  
            # Preprocess image for TFLite  
            input_data = self.preprocess_image(image)  
            
            # Set input tensor  
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)  
            
            # Run inference  
            self.interpreter.invoke()  
            
            # Get output tensors  
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])  
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])  
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])  
            num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])  
            
            # Process detections  
            detections = []  
            for i in range(int(num_detections[0])):  
                score = scores[0][i]  
                if score > self.confidence_threshold:  
                    # Get bounding box coordinates  
                    bbox = boxes[0][i]  
                    class_id = int(classes[0][i])  
                    
                    # Create detection dictionary  
                    detection = {  
                        'box': bbox,  
                        'score': float(score),  
                        'class_id': class_id,  
                        'class_name': self.class_names.get(class_id, 'Unknown')  
                    }  
                    detections.append(detection)  
                    
                    # Draw bounding box  
                    ymin, xmin, ymax, xmax = bbox  
                    h, w = image.shape[:2]  
                    
                    # Convert to pixel coordinates  
                    left = int(xmin * w)  
                    top = int(ymin * h)  
                    right = int(xmax * w)  
                    bottom = int(ymax * h)  
                    
                    # Draw rectangle  
                    cv2.rectangle(image, (left, top), (right, bottom), boxColor, boxLineWidth)  
                    
                    # Annotate  
                    label = f"{self.class_names.get(class_id, 'Unknown')}: {score*100:.1f}%"  
                    cv2.putText(image, label, (left, bottom+10),   
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, boxColor, 2)  
            
            # Show results  
            cv2.imshow('Object Detection', image)  
            cv2.waitKey(1)  
            
            return detections  
        
        except Exception as e:  
            logger.error(f"Detection error: {e}")  
            logger.error(traceback.format_exc())  
            return []  

def parse_arguments():  
    """Parse command-line arguments"""  
    parser = argparse.ArgumentParser(description='Raspberry Pi Object Detection Script')  
    parser.add_argument(  
        '--model',   
        type=str,   
        default=os.path.join(BASE_DIR, 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'),  
        help='Path to TFLite model file'  
    )  
    parser.add_argument(  
        '--confidence',   
        type=float,   
        default=0.4,  
        help='Confidence threshold for detections'  
    )  
    parser.add_argument(  
        '--output',   
        type=str,   
        default=os.path.join(BASE_DIR, 'detected_objects.jpg'),  
        help='Output image filename'  
    )  
    parser.add_argument(  
        '--continuous',   
        action='store_true',  
        help='Run in continuous detection mode'  
    )  
    parser.add_argument(  
        '--interval',   
        type=float,   
        default=1.0,  
        help='Interval between detections in continuous mode (seconds)'  
    )  
    return parser.parse_args()  

def main():  
    """Main execution function with improved signal handling"""  
    global running  
    
    try:  
        # Parse command-line arguments  
        args = parse_arguments()  
        
        # Configure logging  
        logging.getLogger().setLevel(logging.INFO)  
        
        # Log start of execution  
        logger.info("===== OBJECT DETECTION STARTED =====")  
        logger.info(f"Model Path: {args.model}")  
        logger.info(f"Confidence Threshold: {args.confidence}")  
        
        # Create object detection instance  
        detector = RaspberryPiObjectDetection(  
            model_path='ssd_mobilenet_v2_coco_quant_postprocess.tflite',   
            model_labels='coco_labels.txt',  
            confidence_threshold=0.6  
        )  
        
        # Continuous or single detection mode  
        detection_count = 0  
        try:  
            while running:  
                # Capture image  
                image = detector.capture_image()  
                
                if image is None:  
                    logger.error("No image captured.")  
                    break  
                
                # Detect objects  
                detections = detector.detect_objects(image)  
                
                # Optional: print detections  
                for det in detections:  
                    print(f"Detected: {det['class_name']} (Confidence: {det['score']:.2f})")  

                # Wait between detections  
                time.sleep(args.interval)  
        
        except KeyboardInterrupt:  
            print("\nDetection interrupted by user.")  
        
        finally:  
            # Cleanup  
            cv2.destroyAllWindows()  
            logger.info("===== OBJECT DETECTION COMPLETED =====")  
    
    except Exception as e:  
        logger.error(f"Execution failed: {e}")  
        logger.error(traceback.format_exc())  
        sys.exit(1)  

if __name__ == "__main__":  
    main()