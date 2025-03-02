# Copyright 2019 Google LLC  
#  
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
#  
#     https://www.apache.org/licenses/LICENSE-2.0  
#  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.  

"""  
Object detection demo with optimizations for Raspberry Pi.  
"""  

from pycoral.utils.edgetpu import make_interpreter  
import numpy as np  
import time  
import cv2  
from picarx import Picarx  

px = Picarx()  

def _read_label_file(file_path):  
    """Read labels.txt file provided by Coral website."""  
    with open(file_path, 'r', encoding="utf-8") as f:  
        lines = f.readlines()  
    ret = {}  
    for line in lines:  
        pair = line.strip().split(maxsplit=1)  
        ret[int(pair[0])] = pair[1].strip()  
    return ret  

def move_car(direction):  
    """Move the car based on the ball's position."""  
    if direction == "top" :  
        px.set_dir_servo_angle(0)  
        px.forward(2)  
        print(f"Direction {direction}. Moving forward")  
    elif direction == "left":  
        px.set_dir_servo_angle(-30)  
        px.forward(2)  
        print(f"Direction {direction}. Moving left")  
    elif direction == "right":  
        px.set_dir_servo_angle(30)  
        px.forward(2)  
        print(f"Direction {direction}. Moving right")  
    else:  
        print(f"Direction {direction}. No action")  

def locate_object(top_x, top_y, image_width, image_height, margin=0.4):  
    """Determine the position of an object in the image."""  
    if top_x < margin * image_width:  
        return "left"  
    elif top_x > (1 - margin) * image_width:  
        return "right"  
    elif top_y < margin * image_height:  
        return "top"  
    elif top_y > (1 - margin) * image_height:  
        return "bottom"  
    else:  
        return "center"  

def main():  
    # Model and label paths  
    model_filename = "ssd_mobilenet_v2_coco_quant_postprocess.tflite"  
    label_filename = "coco_labels.txt"  

    # Initialize interpreter and labels  
    interpreter = make_interpreter(model_filename)  
    interpreter.allocate_tensors()  
    labels = _read_label_file(label_filename)  

    # Camera settings  
    CAMERA_WIDTH = 1280  
    CAMERA_HEIGHT = 720  
    gst_pipeline = (  
        f"libcamerasrc ! "  
        f"video/x-raw, width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, framerate=10/1 ! "  
        f"videoconvert ! "  
        f"video/x-raw, format=BGR ! "  
        f"appsink"  
    )  

    # Initialize GStreamer capture  
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)  
    if not cap.isOpened():  
        print("Failed to open GStreamer pipeline")  
        return  

    # Frame skipping and delay settings  
    frame_skip = 1  # Process every 3rd frame  
    frame_counter = 0  

    try:  
        while True:  
            # Capture frame from the camera  
            ret, frame = cap.read()  
            if not ret:  
                print("Failed to capture frame")  
                break  

            frame_counter += 1  
            if frame_counter % frame_skip != 0:  
                continue  # Skip this frame  

            # Preprocess the image for the model  
            input_shape = interpreter.get_input_details()[0]['shape']  
            resized_image = cv2.resize(frame, (input_shape[1], input_shape[2]))  
            input_data = resized_image.astype(np.uint8)  
            input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)  

            # Set input tensor  
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)  

            # Run inference  
            interpreter.invoke()  

            # Get output tensors  
            boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])  
            classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])  
            scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])  
            num_detections = int(interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0])  

            # Process detections  
            position = None  
            for i in range(num_detections):  
                class_id = int(classes[0][i])  
                if class_id == 36 and scores[0][i] > 0.05:  # Confidence threshold  
                    h, w = frame.shape[:2]  
                    ymin, xmin, ymax, xmax = boxes[0][i]  
                    left = int(xmin * w)  
                    bottom = int(ymax * h)  

                    position = locate_object(left, bottom, w, h)  
                    print(f"The object is around the {position} of the image.")  

                    # Move the car based on the ball's position  
                    if position != "center":  
                        move_car(position)  
                    else:  
                        px.forward(0)  # Stop the car if the ball is in the center  

            # Add a small delay to reduce CPU usage  
            time.sleep(0.1)  
            px.set_dir_servo_angle(0)  
            px.forward(0)  

    finally:  
        # Release the camera and close OpenCV windows  
        cap.release()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()