import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import sys
import signal
from datetime import datetime
from ultralytics import YOLO

# The PyCoral library helps us use TensorFlow Lite models on Coral EdgeTPU devices
from pycoral.utils.edgetpu import make_interpreter

# A helper library for controlling a PiCar-X
from picarx import Picarx

# The Kalman Filter library from FilterPy
from filterpy.kalman import KalmanFilter

import threading
import queue

# ======================= GLOBAL CONSTANTS & INITIAL SETUP =======================

# Maximum frames we allow to lose track of the ball before we assume it's lost.
MAX_MISSING_FRAMES = 2  

# Some search mode constants (for future expansions or references)
SEARCH_MODE_FRAMES = 4
SEARCH_ANGLE = 15  
SEARCH_SPEED = 1  

# Time-step constants
DT_MAX = 0.4
DT_DEFAULT = 0.2

# Initialize the PiCar-X hardware
px = Picarx()

# Camera resolution
#CAMERA_WIDTH = 640
#CAMERA_HEIGHT = 360
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

# Set up the camera (using V4L2 to control parameters more directly)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, 8)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cv2.waitKey(1000)  # A small pause to let camera settings take effect

# This event signals threads to stop when we're done
stop_event = threading.Event()

def signal_handler(sig, frame):
    """
    A signal handler to catch Ctrl+C (SIGINT), then cleanly:
    - Stop the car
    - Release camera
    - Close any windows
    - Stop threads
    """
    px.stop()
    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ======================== IMAGE PROCESSOR THREAD =========================
class ImageProcessorThread():
    """
    Captures raw frames from the camera in one thread,
    then processes them (undistortion + contrast enhancement) in another thread.
    
    This threading allows us to:
    1) Capture from camera continuously
    2) Transform frames (like undistorting a fisheye lens)
    3) Provide a queue of "ready" frames for the inference thread
    """
    def __init__(self):
        super().__init__()
        
        self.cap = cap
        # We'll store the raw frames in one queue
        self.frame_queue = queue.Queue(maxsize=1)
        
        # We'll store the processed frames in another queue
        self.processed_queue = queue.Queue(maxsize=1)
        
        # Stop event to end the threads gracefully
        self.stop_event = stop_event
        
        # We create two threads:
        # 1) capture_thread gets raw frames from camera
        # 2) transform_thread applies undistort & contrast
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.transform_thread = threading.Thread(target=self.transform_frames)
        
        # CLAHE used for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2))

    def undistort_fisheye(self, image):
        """
        Removes (approx) fisheye distortion from the camera.
        Then crops the edges to reduce artifacts.
        """
        h, w = image.shape[:2]
        
        # K is the camera matrix, D is distortion params
        K = np.array([[w / 1.5, 0,     w / 2],
                      [0,       w/1.5, h / 2],
                      [0,       0,     1    ]],
                     dtype=np.float32)
        D = np.array([-0.3, 0.1, 0, 0], dtype=np.float32)
        
        # Estimate a new camera matrix for undistortion
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3)
        )
        
        # Build remap
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        
        # Apply the remap to undistort
        undistorted = cv2.remap(
            image, map1, map2, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        # Crop 10% edges top, bottom, left, right
        cropped = undistorted[int(h*0.1):int(h*0.9),
                              int(w*0.1):int(w*0.9)]
        
        return cropped

    def enhance_contrast(self, image):
        """
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to enhance the brightness/contrast of the L-channel in LAB space.
        """
        uimg = cv2.UMat(image)
        lab = cv2.cvtColor(uimg, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE on the L (lightness) channel
        l = self.clahe.apply(l)

        # Recombine channels, convert back to BGR
        merged_lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return enhanced.get()  # Return a normal numpy array

    def capture_frames(self):
        """
        Continuously capture frames from the camera. 
        Store only the latest frame in the 'frame_queue'.
        """
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Attempt to place the new frame in the queue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # If it's full, remove the old frame
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                # Then put the new one
                self.frame_queue.put_nowait(frame)
            
            # Small sleep to reduce CPU usage
            time.sleep(0.01)

    def transform_frames(self):
        """
        Fetch frames from 'frame_queue', apply undistortion + contrast
        Then put them into 'processed_queue'.
        """
        start_time = time.time()
        frame_count = 0
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get()
            except queue.Empty:
                continue
            
            # Apply transformations
            frame = self.undistort_fisheye(frame)
            frame = self.enhance_contrast(frame)

            # Put the processed frame into the processed_queue
            try:
                self.processed_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.processed_queue.get_nowait()
                except queue.Empty:
                    pass
                self.processed_queue.put_nowait(frame)

            # Print out frames per second for debugging
            end_time = time.time()
            frame_count += 1
            fps = frame_count / (end_time - start_time)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                  f"FPS Image Capture and Transform: {fps:.3f}")

            # Another small sleep to reduce CPU usage
            time.sleep(0.01)

    def start(self):
        """Start both threads."""
        self.capture_thread.start()
        self.transform_thread.start()

    def stop(self):
        """
        Signal the threads to stop and wait for them to finish.
        Also release the camera resource.
        """
        self.stop_event.set()
        self.capture_thread.join()
        self.transform_thread.join()
        self.cap.release()

# =========================== BALL TRACKER (PID) ============================
class BallTracker:
    """
    Uses a basic PID (Proportional, Integral, Derivative) controller to:
    1) STEER the car left/right to keep the ball horizontally centered
    2) CONTROL SPEED forward/backward to keep the ball at the correct distance
    
    This is a typical approach in robotics to maintain a target position.
    Here, the "target" is the center of our camera for steering,
    and a desired 'distance' to the ball for speed.
    """

    def __init__(self, img_width=1280, img_height=720,
                 steering_limit=35, min_speed=1, max_speed=4):
        self.img_width = img_width
        self.img_height = img_height

        # Center in x,y on the image
        self.center_x = img_width // 2
        self.center_y = img_height // 2

        # Steering PID memory
        self.prev_steering_error = 0
        self.steering_integral = 0

        # Speed PID memory
        self.prev_distance_error = 0
        self.distance_integral = 0

        # Limits and speed range
        self.steering_limit = steering_limit
        self.min_speed = min_speed
        self.max_speed = max_speed

        # PID Gains for Steering
        self.kp_steering = 0.5  
        self.ki_steering = 0.03  
        self.kd_steering = 0.15  

        # PID Gains for Speed
        self.kp_speed = 0.07  
        self.ki_speed = 0.01  
        self.kd_speed = 0.1  

        # Some binning sizes (not strongly used but might help in older code)
        self.steering_bin_size = 20  
        self.speed_bin_size = self.img_height * 0.05  

    def get_steering_and_speed(self, ball_coords):
        """
        Given the bounding box of the ball in image coordinates,
        compute:
        1) The STEERING angle (left/right) as a PID controlling horizontal offset
        2) The SPEED (forward) as a PID controlling the distance to the ball
        """
        
        # Unpack bounding box: (left, top, right, bottom)
        left, top, right, bottom = ball_coords

        # Calculate the center of the ball in x,y
        ball_x = (left + right) // 2  
        ball_y = (top + bottom) // 2  

        # ================== STEERING CALC (PID) =====================
        steering_error = ball_x - self.center_x

        # Normalize the error so it roughly ranges -1..1 if the ball is in view
        normalized_steering_error = (steering_error / self.center_x) * 2

        # Scale up for large errors so we steer more aggressively
        correction_factor = 1.0 + min(abs(normalized_steering_error)*1.5, 2)

        # PID components
        self.steering_integral += normalized_steering_error
        steering_derivative = normalized_steering_error - self.prev_steering_error

        P_steering = self.kp_steering * normalized_steering_error * correction_factor
        I_steering = self.ki_steering * self.steering_integral
        D_steering = self.kd_steering * steering_derivative

        # The raw steering angle
        steering_angle = P_steering + I_steering + D_steering

        self.prev_steering_error = normalized_steering_error

        # Further factor in how far the ball is vertically
        # (This code tries to steer more if ball is far up the image)
        distance_factor = 1.0 + min((self.img_height - ball_y) / self.img_height, 0.55)
        correction_factor *= distance_factor
        steering_angle *= correction_factor

        # Clip to the max turning radius
        steering_angle = np.clip(
            steering_angle, -self.steering_limit, self.steering_limit
        )

        # ===================== SPEED CALC (PID) =====================
        ball_height = bottom - top
        close_threshold = self.img_height * 0.25
        distance_error = (180 - ball_height) // self.speed_bin_size  

        # If the ball is above a certain size in image => it is "close"
        if ball_height >= close_threshold:
            # Minimum speed
            speed = self.min_speed
            self.distance_integral = 0
        else:
            # Normal PID approach
            self.distance_integral += distance_error
            distance_derivative = distance_error - self.prev_distance_error

            P_speed = self.kp_speed * distance_error
            I_speed = self.ki_speed * self.distance_integral
            D_speed = self.kd_speed * distance_derivative

            speed = P_speed + I_speed + D_speed
            self.prev_distance_error = distance_error

            # If we need a big steering correction, reduce speed
            speed_reduction_factor = max(
                0.5, 1 - abs(normalized_steering_error)*0.7
            )
            speed *= speed_reduction_factor

        # Stop if the ball is big enough => implying very close 
        if ball_height > (self.img_height / 9):
            speed = 0  # Full stop

        # Keep speed within [0..max_speed], unless it's set >0 => min_speed
        speed = np.clip(
            speed, self.min_speed if speed > 0 else 0, self.max_speed
        )

        # We add a 7-degree offset to compensate for camera mount angle
        steering_angle += 7

        # Debug print to see the final steering angle + speed
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
              f"Steering: {steering_angle:.2f}, Speed: {speed:.2f}")

        return steering_angle, speed

# =========================== KALMAN FILTER ==============================
class KalmanTracker:
    """
    This class 'tracks' the ball for short periods (up to 2 frames)
    when the camera detection fails. 
    A Kalman Filter can PREDICT the ball's next position based on
    velocity and friction if detection is temporarily missing.
    
    If detection reappears, we CORRECT the filter with the real measurement.
    """

    def __init__(self, friction=0.98):
        # 8 states => (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x = np.zeros((8, 1))  # initial state is uncertain
        self.kf.P = np.eye(8) * 200   # we begin with high uncertainty

        self.friction = friction
        self.dt = 0.2
        self.tracking_active = False
        self.missing_counter = 0

        # State transition matrix includes friction & dt
        self.kf.F = np.array([
            [1, 0, 0, 0, self.dt, 0,     0,     0],
            [0, 1, 0, 0, 0,     self.dt, 0,     0],
            [0, 0, 1, 0, 0,     0,      self.dt,0],
            [0, 0, 0, 1, 0,     0,      0,     self.dt],
            [0, 0, 0, 0, self.friction, 0,     0,     0],
            [0, 0, 0, 0, 0,     self.friction,0,     0],
            [0, 0, 0, 0, 0,     0,      self.friction,0],
            [0, 0, 0, 0, 0,     0,      0,     self.friction]
        ])

        # Measurement matrix => we measure x1, y1, x2, y2
        self.kf.H = np.eye(4, 8)

        # R = measurement noise
        self.kf.R = np.eye(4) * 10

        # Q = process noise
        self.kf.Q = np.eye(8) * 5

    def update_transition_matrix(self, dt):
        """
        We'll recalc 'dt' each frame (so if the loop runs slower or faster, 
        the filter uses that actual time step).
        """
        dt = min(dt, 0.4)
        self.kf.F[:4, 4:] = np.eye(4) * dt
        print(f"[Kalman] Updated transition matrix with dt = {dt:.2f}")

    def reinitialize_filter(self, box):
        """
        When the ball reappears after being lost, we reinit:
        - State set to the new detection
        - Keep half the old velocity (just in case)
        - Reset the covariance
        - Mark tracking as active
        """
        x1, y1, x2, y2 = box
        
        vx1, vy1, vx2, vy2 = self.kf.x[4:].flatten() * 0.5

        self.kf.x = np.array([
            [x1], [y1], [x2], [y2],
            [vx1], [vy1], [vx2], [vy2]
        ]).reshape(8, 1)

        self.kf.P = np.eye(8) * 10
        self.tracking_active = True
        self.missing_counter = 0
        print("[Kalman] Ball reappeared, reinitializing with past velocity.")

    def correct_with_measurement(self, box):
        """
        If we have a real measurement from the camera, correct the filter
        with the new bounding box => (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = box
        z = np.array([[x1], [y1], [x2], [y2]], dtype=float)

        self.kf.update(z)
        self.missing_counter = 0

    def predict(self):
        """
        If the detection is missing for a frame or two, we
        do a Kalman 'predict' step to guess ball's location.
        
        Then we do an additional velocity push to simulate motion
        (scaled by friction each step).
        """
        if self.tracking_active:
            self.kf.predict()
        
        # This code attempts to project bounding box forward by velocity
        velocity = self.kf.x[4:].flatten()
        
        # Expand bounding box by dt * velocity, scaled by 1.4
        # (this can be tuned if predictions are too big)
        self.kf.x[:4] += (velocity[:4] * self.dt * 1.4).reshape(-1, 1)

        # Multiply velocity by friction factor
        self.kf.x[4:] *= 0.97

        print(f"[Kalman] Extrapolated with velocity: {self.kf.x[:4].flatten()}")
        return self.kf.x[:4].flatten()

# =========================== INFERENCE THREAD ============================
class InferenceThread():
    """
    Runs the object detection model (TFLite + Coral) on processed frames
    in a separate thread. This helps avoid blocking the main pipeline.
    """
    def __init__(self, processed_queue, output_queue,
                 #model_path="ssd_mobilenet_v2_coco_quant_postprocess.tflite"):
                 model_path="yolo11n_full_integer_quant_320_edgetpu.tflite"):
                 #model_path="yolo11n_full_integer_quant_640_edgetpu.tflite"):
                 #model_path="yolov8n_full_integer_quant_edgetpu.tflite"):
                 #model_path="yolo11s_full_integer_quant_320_edgetpu.tflite"):
        super().__init__()
        self.processed_queue = processed_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

        self.model = YOLO(model_path)
        # This loads the TFLite model for detection
                

    def inferWork(self):
        """
        Continuously fetch frames from 'processed_queue',
        run inference to find the ball (class 36),
        and put the bounding box result in 'output_queue'.
        """
        start_time = time.time()
        frame_count = 0
        
        while not self.stop_event.is_set():
            try:
                print ("starting inference...????")
                frame = self.processed_queue.get()
            except queue.Empty:
                continue
            
            print ("starting inference...")
            results = self.model.predict(frame, imgsz=CAMERA_WIDTH)
            
            print(f"num detections {len(results)}")
            # We'll store the bounding box of the ball in 'box' if found
            ballbox = None
            # Parse the results  
            for result in results:  
                # Access bounding boxes  
                objects = result.boxes  # Boxes object for bounding box outputs  
                for object in objects:  
                    class_id = object.cls.item()  
                    if (class_id  != 32):
                        continue
                    # Get coordinates (x1, y1, x2, y2)  
                    box = object.xyxy[0].tolist()  # Convert tensor to list  
                    # Get confidence score  
                    confidence =object.conf.item()  
                    # Get class ID  
                    class_id = object.cls.item()  
                    # Get class name (if available)  
                    class_name = self.model.names[class_id]  

                    print(f"Detected {class_name}, {class_id} with confidence {confidence:.2f} at {box}")  
                    ballbox = box
                            
            # Attempt to put the detection result (box or None) in the output queue
            try:
                self.output_queue.put_nowait(ballbox)
            except queue.Full:
                # If the queue is full, discard the old item
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                self.output_queue.put_nowait(ballbox)

            frame_count += 1
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                  f"FPS Inference: {fps:.3f}")

    def start(self):
        """Launch the separate thread to run inference forever."""
        thread = threading.Thread(target=self.inferWork)
        thread.start()
        
    def stop(self):
        """Set the stop_event so the inferWork loop ends."""
        self.stop_event.set()

class HSVInferenceThread():
    """
    Runs the object detection model (TFLite + Coral) on processed frames
    in a separate thread. This helps avoid blocking the main pipeline.
    """
    def __init__(self, processed_queue, output_queue):
        super().__init__()
        self.processed_queue = processed_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def inferWork(self):
        """
        Continuously fetch frames from 'processed_queue',
        run inference to find the ball (class 36),
        and put the bounding box result in 'output_queue'.
        """
        start_time = time.time()
        frame_count = 0
        
        while not self.stop_event.is_set():
            try:
                print ("starting inference...????")
                frame = self.processed_queue.get()
            except queue.Empty:
                continue
            
            # 1) Convert from BGR to HSV color space.
            #    HSV makes it easier to isolate specific colors (like "tennis ball" yellow).
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 2) Define the lower and upper HSV bounds for "tennis ball yellow."
            #    These ranges may need to be tweaked depending on your lighting.
            lower_yellow = np.array([15, 82, 67], dtype=np.uint8)
            upper_yellow = np.array([84, 132, 168], dtype=np.uint8)

            # 3) Create a mask that highlights only the yellow pixels in the image.
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 4) Clean up the mask using morphological operations:
            #    - Erode small white spots.
            #    - Dilate the remaining areas, so they grow back slightly.
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # 5) Find the contours (edges/outlines) of all shapes in the mask.
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Variables to keep track of the "best" contour (i.e., the actual tennis ball).
            best_box = None    # This will be (x, y, w, h)
            best_radius = 0.0  # We'll still measure radius to check circularity

            # 6) Loop over each contour to see if it meets our criteria for a tennis ball.
            for cnt in contours:
                area = cv2.contourArea(cnt)  # The area (in pixels) of the contour
                
                # Get the minimum enclosing circle (center & radius)
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                
                # Compute circularity:
                #   area of contour / area of circle with the same radius
                circle_area = np.pi * (radius ** 2) if radius > 0 else 1  # avoid divide-by-zero
                circularity = area / circle_area

                # Filter by area (not too small/large) AND by circularity (close to 1 means "circle").
                if 100 < area < 20000 and 0.2 < circularity < 2:
                    # If it passes the test, get the bounding box (rectangle) for the contour
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Check if this contour is better (larger radius) than our current best
                    if radius > best_radius:
                        best_radius = radius
                        best_box = (x, y, w, h)

            # Attempt to put the detection result (box or None) in the output queue
            try:
                self.output_queue.put_nowait(best_box)
            except queue.Full:
                # If the queue is full, discard the old item
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                self.output_queue.put_nowait(best_box)

            frame_count += 1
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                  f"HSV FPS Inference: {fps:.3f}")

    def start(self):
        """Launch the separate thread to run inference forever."""
        thread = threading.Thread(target=self.inferWork)
        thread.start()
        
    def stop(self):
        """Set the stop_event so the inferWork loop ends."""
        self.stop_event.set()

# ========================== MAIN CAR CONTROLLER ==========================
class CarController:
    """
    Orchestrates the entire pipeline:
    1) Starts the camera capture & transformation threads
    2) Starts the inference thread
    3) Retrieves bounding boxes from the detection queue
    4) Applies Kalman filtering if the box is missing (<=2 frames)
    5) Uses BallTracker (PID) to compute steering & speed
    6) Sends commands to the PiCar-X hardware
    """
    def __init__(self, picar):
        self.px = picar  # The PiCar-X interface

        # The Kalman tracker to handle short missing detections
        self.tracker = KalmanTracker()

        # The PID-based Ball Tracker for controlling steering & speed
        self.ballTracker = BallTracker(CAMERA_WIDTH, CAMERA_HEIGHT)

        # Image processor: fetch & transform frames
        self.image_processor = ImageProcessorThread()

        # A small queue for detection results
        self.detection_queue = queue.Queue(maxsize=1)

        # Start the TFLite-based inference in a separate thread
        self.inference_thread = InferenceThread(
            self.image_processor.processed_queue,
            self.detection_queue
        )

        # Launch the threads
        self.image_processor.start()
        self.inference_thread.start()

        self.missing_frames = 0
        self.search_mode = False
        self.search_direction = 1

    def move(self, steering_angle, speed):
        """
        Actually send commands to PiCar-X:
        1) steering_angle (left/right)
        2) speed (forward)
        """
        self.px.set_dir_servo_angle(steering_angle)
        self.px.forward(speed)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
              f"Moved car: Steering {steering_angle}, Speed {speed}")

    def processPipeline(self):
        """
        Main loop that runs forever:
        - Wait for a detection from 'detection_queue'
        - If we get a bounding box (box != None), we trust it and correct Kalman
        - If box == None, we see if Kalman can predict for up to 2 frames
        - If ball missing >2 frames, we stop tracking & set speed=0
        - Then we pass final_box to BallTracker => get steering & speed => move
        """
        prev_time = time.time()
        print("Starting pipeline...")

        try:
            while True:
                # Wait for the latest detection (blocking get)
                box = self.detection_queue.get()

                now = time.time()
                dt = now - prev_time
                prev_time = now

                if box is not None:
                    # We have a real detection => reset missing_frames
                    self.missing_frames = 0
                    self.tracker.update_transition_matrix(dt)

                    if not self.tracker.tracking_active:
                        self.tracker.reinitialize_filter(box)
                    else:
                        self.tracker.correct_with_measurement(box)

                    final_box = box
                else:
                    # No detection => maybe use Kalman to predict
                    self.missing_frames += 1

                    if (self.tracker.tracking_active and 
                        self.missing_frames <= MAX_MISSING_FRAMES):
                        predicted_box = self.tracker.predict()
                        final_box = predicted_box
                    else:
                        # We lost the ball beyond 2 frames => stop or search
                        print("[Kalman] Ball missing too long, stopping tracking.")
                        self.tracker.tracking_active = False
                        final_box = None

                # If we do have a final box, control the car
                if final_box is not None:
                    # final_box is [x1, y1, x2, y2]
                    steering_angle, speed = self.ballTracker.get_steering_and_speed(final_box)
                    if speed > 0:
                        self.move(steering_angle, speed)
                    else:
                        # If speed <= 0, we just stop
                        self.move(0, 0)
                else:
                    # No box => we can't track => stop
                    self.move(0, 0)

        except KeyboardInterrupt:
            # If user hits Ctrl+C
            print("Process interrupted by user.")
            self.move(0, 0)

# =========================== MAIN ENTRYPOINT ============================
def main():
    """
    The main entrypoint that creates a CarController
    and starts its pipeline loop forever.
    """
    print("Starting car tracking system...")
    carController = CarController(px)
    carController.processPipeline()

# Standard Python check to run main
if __name__ == "__main__":
    main()
