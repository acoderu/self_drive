from pycoral.utils.edgetpu import make_interpreter  
import numpy as np  
import time  
import cv2  
from picarx import Picarx  
from filterpy.kalman import KalmanFilter  

# Initialize the Picarx car  
px = Picarx()  

# Initialize Kalman Filter for tracking the ball  
kf = KalmanFilter(dim_x=4, dim_z=2)  
kf.x = np.array([[0], [0], [0], [0]])  
kf.P *= 1000.  
kf.F = np.array([[1, 0, 1, 0],  
                 [0, 1, 0, 1],  
                 [0, 0, 1, 0],  
                 [0, 0, 0, 1]])  
kf.H = np.array([[1, 0, 0, 0],  
                 [0, 1, 0, 0]])  
kf.R = np.array([[10, 0],  
                 [0, 10]])  
kf.Q = np.array([[1, 0, 0, 0],  
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

def move_car(steering_angle, speed=10):  
    """Move the car based on the calculated steering angle and speed."""  
    px.set_dir_servo_angle(steering_angle)  
    px.forward(speed)  
    print(f"Moving with steering angle {steering_angle} at speed {speed}")  

def proportional_control(ball_x, ball_y, image_width, image_height):  
    """Calculate the steering angle using proportional control."""  
    center_x, center_y = image_width / 2, image_height / 2  
    error_x = ball_x - center_x  # Horizontal error  

    # Proportional gain for steering  
    kp = 0.1  # Adjust this value for smoother or sharper turns  
    steering_angle = -kp * error_x  # Negative sign to correct direction  

    # Limit steering angle to avoid jerky movements  
    steering_angle = max(min(steering_angle, 30), -30)  
    return steering_angle  

def avoid_obstacle():  
    """Avoid an obstacle by steering left or right."""  
    # Check the distance to the obstacle  
    distance = round(px.ultrasonic.read(), 2)  
    if distance < 20:  # Obstacle detected within 20 cm  
        print(f"Obstacle detected at {distance} cm. Avoiding...")  
        # Steer left or right to avoid the obstacle  
        px.set_dir_servo_angle(30)  # Steer right  
        px.forward(10)  # Move forward at a slow speed  
        time.sleep(1)  # Wait for 1 second  
        px.set_dir_servo_angle(-30)  # Steer left  
        px.forward(10)  # Move forward at a slow speed  
        time.sleep(1)  # Wait for 1 second  
        px.set_dir_servo_angle(0)  # Reset steering angle  
        px.forward(0)  # Stop the car  
        return True  # Obstacle avoided  
    return False  # No obstacle detected  

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

                    # Check for obstacles and avoid them if necessary  
                    if not avoid_obstacle():  
                        # Calculate steering angle using proportional control  
                        steering_angle = proportional_control(estimated_x, estimated_y, w, h)  
                        move_car(steering_angle)  # Move the car towards the ball  

            # If the ball is not found, stop the car  
            if not ballFound:  
                px.forward(0)  

    finally:  
        # Release the camera and close OpenCV windows  
        cap.release()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()