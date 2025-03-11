from pycoral.utils.edgetpu import make_interpreter  
import numpy as np  
import time  
import cv2  
from picarx import Picarx  
from filterpy.kalman import KalmanFilter  # Import Kalman Filter  

# Initialize the Picarx car  
px = Picarx()  

# 1. Kalman Filter: Predicting the Ball’s Position
# What is the Kalman Filter?
# The Kalman Filter is like a "smart predictor." It helps us estimate the position of the ball even when our measurements (from the camera) are noisy or incomplete. It combines:

# Predictions: Where the ball is likely to be based on its previous position and velocity.
# Measurements: Where the ball is detected in the current frame.
# By blending these two, the Kalman Filter gives us a smooth and accurate estimate of the ball’s position.

# Key Components of the Kalman Filter
# State Transition:

# This is how the ball’s position and velocity are expected to change over time.
# Example: If the ball is moving to the right at 10 pixels per frame, the Kalman Filter predicts its next position based on this velocity.
# Mathematically, this is represented by the state transition matrix (kf.F).
# Measurement Noise:

# This represents the uncertainty in the camera’s detection of the ball.
# Example: If the camera sometimes detects the ball a few pixels off its actual position, this is accounted for as measurement noise.
# Mathematically, this is represented by the measurement noise matrix (kf.R).
# Process Noise:

# This represents the uncertainty in our prediction of the ball’s movement.
# Example: If the ball might accelerate or decelerate unpredictably, this is accounted for as process noise.
# Mathematically, this is represented by the process noise matrix (kf.Q).
# How the Kalman Filter Works
# Predict:

# The Kalman Filter predicts the ball’s next position based on its current position and velocity.
# Example: If the ball is at position (100, 200) and moving at 10 pixels per frame to the right, it predicts the next position as (110, 200).
# Update:

# The Kalman Filter updates its prediction using the actual measurement from the camera.
# Example: If the camera detects the ball at (115, 200), the Kalman Filter combines this with its prediction to give a more accurate estimate.
# Blend:

# The Kalman Filter blends the prediction and measurement, giving more weight to the one with less uncertainty.
# Example: If the camera is very noisy, the Kalman Filter relies more on its prediction. If the camera is accurate, it relies more on the measurement.
# 2. Proportional Control: Smooth Steering
# What is Proportional Control?
# Proportional Control is like a "smart driver" that adjusts the car’s steering based on how far the ball is from the center of the frame. It works by:

# Calculating the error: How far the ball is from the center.
# Adjusting the steering angle proportionally to the error.
# How Proportional Control Works
# Calculate the Error:

# The error is the difference between the ball’s position and the center of the frame.
# Example: If the ball is at (200, 300) and the center is at (320, 240), the error in the x-direction is 200 - 320 = -120.
# Adjust the Steering Angle:

# The steering angle is calculated as steering_angle = -kp * error, where kp is the proportional gain.
# Example: If kp = 0.1 and the error is -120, the steering angle is -0.1 * -120 = 12°.
# Limit the Steering Angle:

# To prevent jerky movements, the steering angle is limited to a maximum value (e.g., 30°).
# Why Proportional Control is Smooth
# Small Error: If the ball is close to the center, the steering angle is small, so the car moves gently.
# Large Error: If the ball is far from the center, the steering angle is larger, so the car turns more sharply.
# No Sudden Changes: The steering angle changes gradually, making the car’s movement smooth.
# 3. Combining Kalman Filter and Proportional Control
# How They Work Together
# Kalman Filter:

# Predicts the ball’s position and smooths out noisy measurements.
# Provides a reliable estimate of where the ball is and where it’s going.
# Proportional Control:

# Uses the ball’s estimated position to calculate the steering angle.
# Adjusts the car’s movement smoothly to keep the ball centered.
# Example Scenario
# Ball Moving Right:

# The Kalman Filter predicts the ball’s position slightly to the right of the center.
# Proportional Control calculates a small steering angle to the right, so the car turns gently to follow the ball.
# Ball Near the Car (Stopping Zone):

# The Kalman Filter detects the ball in the stopping zone (lower center of the frame).
# The car stops because the ball is close.
# Ball Moving Left:

# The Kalman Filter predicts the ball’s position slightly to the left of the center.
# Proportional Control calculates a small steering angle to the left, so the car turns gently to follow the ball.

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

def is_ball_near(ball_x, ball_y, image_width, image_height):  
    """Check if the ball is in the stopping zone (lower center of the frame)."""  
    # Define the stopping zone as a rectangle in the lower center of the frame  
    zone_width = image_width * 0.4  # 40% of the frame width  
    zone_height = image_height * 0.2  # 20% of the frame height  
    zone_x1 = int((image_width - zone_width) / 2)  # Left boundary  
    zone_x2 = int((image_width + zone_width) / 2)  # Right boundary  
    zone_y1 = int(image_height - zone_height)  # Top boundary  
    zone_y2 = image_height  # Bottom boundary  

    # Check if the ball is within the stopping zone  
    return (zone_x1 <= ball_x <= zone_x2) and (zone_y1 <= ball_y <= zone_y2)  

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

                    # Check if the ball is near the car (in the stopping zone)  
                    if is_ball_near(estimated_x, estimated_y, w, h):  
                        px.forward(0)  # Stop the car if the ball is near  
                        print("Ball is near. Car stopped.")  
                    else:  
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