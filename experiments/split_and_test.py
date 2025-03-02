#!/usr/bin/env python3  

import cv2  
import numpy as np  
import time  
import logging  
import sys  
import traceback  
import argparse  
import os  
import signal  

# libcamera and Picamera2 imports  
import libcamera  
from picamera2 import Picamera2  
import numpy as np  

# TFLite import  
import tflite_runtime.interpreter as tflite  

# Set the base directory  
BASE_DIR = "/home/usman/tracking"  

# Global variable to control main loop  
running = True  

def signal_handler(signum, frame):  
    """  
    Handle interrupt signals (Ctrl+C)  
    """  
    global running  
    print("\nInterrupt received, shutting down...")  
    running = False  

# Register signal handlers  
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C  
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal  

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
    def __init__(self, model_path, confidence_threshold=0.4):  
        """  
        Initialize the object detection system for Raspberry Pi  
        """  
        logger.info("===== INITIALIZING OBJECT DETECTION =====")  
        
        try:  
            # Camera configuration  
            self.width = 1640  
            self.height = 1232  
            self.confidence_threshold = confidence_threshold  
            
            # Validate model path  
            if not os.path.exists(model_path):  
                logger.error(f"Model file not found: {model_path}")  
                raise FileNotFoundError(f"Model file not found: {model_path}")  
            
            # Initialize Picamera2  
            self.picam2 = Picamera2()  
            
            # Configure camera with advanced settings  
            config = self.picam2.create_still_configuration(  
                main={"size": (self.width, self.height)},  
                lores={"size": (640, 480)}  
            )  
            self.picam2.configure(config)  
            
            # Optional: Set advanced camera controls  
            #try:  
            #    self.picam2.set_controls({  
            #        "AeEnable": True,  
            #        "ExposureTime": 20000,  # microseconds  
            #        "AnalogueGain": 1.0,  
            #        "AeMeteringMode": libcamera.controls.AeMeteringModeEnum.Spot  
            #    })  
            #except Exception as control_err:  
            #    logger.warning(f"Could not set camera controls: {control_err}")  
            
            # Load TFLite model  
            logger.info(f"Loading TFLite model from {model_path}")  
            self.interpreter = tflite.Interpreter(  
                model_path=model_path,  
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', {})]  
            )  
            self.interpreter.allocate_tensors()  

            # Get input and output details  
            self.input_details = self.interpreter.get_input_details()  
            self.output_details = self.interpreter.get_output_details()  

            # Log detailed input and output tensor information  
            self._log_tensor_details()  

            # Load class names  
            self.class_names = self.load_class_names(os.path.join(BASE_DIR, 'coco_labels.txt'))  
            
            logger.info("Initialization complete")  
        
        except Exception as e:  
            logger.error(f"Initialization failed: {e}")  
            logger.error(traceback.format_exc())  
            raise  

    def _log_tensor_details(self):  
        """  
        Log detailed information about input and output tensors  
        """  
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
        """  
        Load class names from a file  
        """  
        try:  
            with open(filename, 'r') as f:  
                return {i: line.strip() for i, line in enumerate(f.readlines())}  
        except Exception as e:  
            logger.error(f"Failed to load class names: {e}")  
            return {}  

    def capture_image(self):  
        """  
        Capture image using Picamera2 with robust error handling  
        """  
        logger.info("===== STARTING IMAGE CAPTURE PROCESS =====")  
        
        try:  
            # Start the camera  
            self.picam2.start()  
            
            # Allow camera to stabilize  
            time.sleep(2)  
            
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
        """  
        Preprocess image for quantized TFLite model  
        """  
        try:  
            # Resize to model input size  
            resized_image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)  
            
            # Get input tensor details  
            input_details = self.input_details[0]  
            
            # Quantization parameters  
            input_scale = input_details.get('quantization', (1.0, 0))[0]  
            input_zero_point = input_details.get('quantization', (1.0, 0))[1]  
            
            # Normalize and quantize  
            normalized = resized_image.astype(np.float32) / 255.0  
            quantized = ((normalized / input_scale) + input_zero_point).astype(np.uint8)  
            
            # Expand dimensions  
            input_tensor = np.expand_dims(quantized, axis=0)  
            
            return input_tensor  
        
        except Exception as e:  
            logger.error(f"Image preprocessing failed: {e}")  
            logger.error(f"Input details: {self.input_details}")  
            raise  

    def detect_objects(self, image):  
        """  
        Detect objects in the image with comprehensive logging  
        """  
        try:  
            # Preprocess image  
            input_image = self.preprocess_image(image)  
            
            # Set input tensor  
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)  
            
            # Run inference  
            self.interpreter.invoke()  
            
            # Get output tensors  
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])  
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])  
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])  
            
            # Process detections  
            detections = []  
            for i in range(len(classes[0])):  
                score = scores[0][i]  
                cls = int(classes[0][i])  
                
                if score > self.confidence_threshold:  
                    detections.append({  
                        'class': cls,  
                        'class_name': self.class_names.get(cls, f'Unknown (Class {cls})'),  
                        'score': float(score),  
                        'box': boxes[0][i]  
                    })  
            
            # Sort by confidence  
            detections.sort(key=lambda x: x['score'], reverse=True)  
            
            return detections  
        
        except Exception as e:  
            logger.error(f"Detection error: {e}")  
            logger.error(traceback.format_exc())  
            return []  

    def visualize_detections(self, image, detections):  
        """  
        Visualize detected objects on the image  
        """  
        output_image = image.copy()  
        height, width = output_image.shape[:2]  
        
        # Color palette with more classes  
        colors = {  
            'person': (255, 0, 0),       # Blue  
            'book': (0, 0, 255),         # Red  
            'sports ball': (0, 255, 0),  # Green  
            'dog': (255, 255, 0),        # Cyan  
            'cat': (255, 0, 255),        # Magenta  
            'default': (128, 128, 128)   # Gray  
        }  
        
        for detection in detections:  
            # Unnormalize coordinates  
            ymin, xmin, ymax, xmax = detection['box']  
            
            # Convert to pixel coordinates  
            xmin = int(xmin * width)  
            xmax = int(xmax * width)  
            ymin = int(ymin * height)  
            ymax = int(ymax * height)  
            
            # Choose color  
            color = colors.get(detection['class_name'], colors['default'])  
            
            # Draw bounding box  
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)  
            
            # Draw label  
            label = f"{detection['class_name']}: {detection['score']:.2f}"  
            cv2.putText(  
                output_image,   
                label,   
                (xmin, ymin - 10),   
                cv2.FONT_HERSHEY_SIMPLEX,   
                0.9,   
                color,   
                2  
            )  
        
        return output_image  

def parse_arguments():  
    """  
    Parse command-line arguments  
    """  
    parser = argparse.ArgumentParser(description='Raspberry Pi Object Detection Script')  
    parser.add_argument(  
        '--model',   
        type=str,   
        default=os.path.join(BASE_DIR, 'ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite'),  
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
    """  
    Main execution function with improved signal handling  
    """  
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
            model_path=args.model,   
            confidence_threshold=args.confidence  
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
                
                # Log detection results  
                logger.info(f"Detection {detection_count + 1}")  
                logger.info(f"Total objects detected: {len(detections)}")  
                for det in detections:  
                    logger.info(f"Detected: {det['class_name']} (Confidence: {det['score']:.2f})")  
                
                # Visualize detections  
                output_image = detector.visualize_detections(image, detections)  
                
                # Save output image with incrementing filename  
                output_filename = os.path.join(  
                    BASE_DIR,   
                    f'detected_objects_{detection_count}.jpg'  
                )  
                cv2.imwrite(output_filename, output_image)  
                logger.info(f"Output image saved to {output_filename}")  
                
                # Display image  
                cv2.imshow('Object Detection', output_image)  
                key = cv2.waitKey(1) & 0xFF  
                if key == 27:  # ESC key  
                    break  
                
                # Increment detection count  
                detection_count += 1  
                
                # If not in continuous mode, break after first detection  
                if not args.continuous:  
                    break  
                
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