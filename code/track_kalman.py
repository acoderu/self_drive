from pycoral.utils.edgetpu import make_interpreter  
import numpy as np  
import time  
import cv2  
from picarx import Picarx  
from filterpy.kalman import KalmanFilter  # Import Kalman Filter  

# Initialize the Picarx car  
px = Picarx()  

# Initialize Kalman Filter for tracking the ball  
kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)  
kf.x = np.array([[0], [0], [0], [0]])  # Initial state (position and velocity)  
kf.P *= 1000.  # Initial uncertainty  
kf.F = np.array([[1, 0, 1, 0],  # State transition matrix  
                 [0, 1, 0, 1],  
                 [0, 0, 1, 0],  
                 [0, 0, 0, 1]])  
kf.H = np.array([[1, 0, 0, 0],  # Measurement function  
                 [0, 1, 0, 0]])  
kf.R = np.array([[10, 0],  # Measurement noise  
                 [0, 10]])  
kf.Q = np.array([[1, 0, 0, 0],  # Process noise  
                 [0, 1, 0, 0],  
                 [0, 0, 1, 0],  
                 [0, 0, 0, 1]])  

def _read_label_file(file_path):  
    """Read labels.txt file provided by Coral website."""  
    with open(file_path, 'r', encoding="utf-8") as f:  
        lines = f.readlines()  
    ret = {}  
    for line in lines:  
        pair = line.strip().split(maxsplit=1)  
        ret[int(pair[0])] = pair[1].strip()  
    return ret  

def move_car(direction, speed=10):  
    """Move the car based on the calculated direction and speed."""  
    if direction == "top":  
        px.set_dir_servo_angle(0)  
        px.forward(speed)  
        print(f"Moving forward at speed {speed}")  
    elif direction == "left":  
        px.set_dir_servo_angle(-30)  
        px.forward(speed)  
        print(f"Moving left at speed {speed}")  
    elif direction == "right":  
        px.set_dir_servo_angle(30)  
        px.forward(speed)  
        print(f"Moving right at speed {speed}")  
    else:  
        px.forward(0)  
        print(f"No action taken. Car stopped.")  

def locate_object(top_x, top_y, image_width, image_height):  
    """Determine the position of an object in the image."""  
    center_x, center_y = image_width / 2, image_height / 2  
    if top_x < center_x and top_y < center_y :
        return "left"  
    elif top_x > center_x and top_y < center_y:
        return "right"  
    elif top_y < center_y:  
        return "top"  
    else:
        return "bottom"
    #elif top_y > center_y:
    #    return "bottom"
    #else:
    #    return "center"

def main():  
    # Model and label paths  
    model_filename = "../models/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
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
        f"video/x-raw, width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, framerate=6/1 ! "  
        f"videoconvert ! "  
        f"video/x-raw, format=BGR ! "  
        f"appsink"  
    )  

    # Initialize GStreamer capture  
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)  
    if not cap.isOpened():  
        print("Failed to open GStreamer pipeline")  
        return  

    # Frame skipping settings  
    frame_skip = 2  
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
            ballFound = False  
            for i in range(num_detections):  
                class_id = int(classes[0][i])  
                if class_id == 36 and scores[0][i] > 0.05:  # Confidence threshold for ball detection  
                    ballFound = True  
                    h, w = frame.shape[:2]  
                    ymin, xmin, ymax, xmax = boxes[0][i]  
                    left = int(xmin * w)  
                    bottom = int(ymax * h)  

                    # Use Kalman Filter to predict the ball's position  
                    kf.predict()  # Predict the next state  
                    kf.update(np.array([[left], [bottom]]))  # Update with the new measurement  

                    # Get the estimated position from the Kalman Filter  
                    estimated_x, estimated_y = kf.x[0], kf.x[1]  

                    # Determine the direction based on the estimated position  
                    position = locate_object(estimated_x, estimated_y, w, h)  
                    print(f"The object is around the {position} of the image.")  

                    # Move the car for (left, right and top) positions
                    if position != 'bottom':
                        move_car(position)  
                    else:  
                        px.forward(0)  # Stop the car if the ball is in the bottom

            # If the ball is not found, stop the car  
            if not ballFound:  
                px.forward(0)  

    finally:  
        # Release the camera and close OpenCV windows  
        cap.release()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()