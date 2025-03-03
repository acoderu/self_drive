#!/usr/bin/env python3
"""
This script uses a pre-trained TensorFlow Lite model (SSD MobileNet V2) on an Edge TPU
to detect a "sports ball" in a camera feed. Once detected, the PicarX robot car will
steer itself to follow the ball based on its position in the camera frame.
"""

# -------------------------
# IMPORT LIBRARIES
# -------------------------
from pycoral.utils.edgetpu import make_interpreter  # For interacting with the Edge TPU
import numpy as np                                   # For numerical computations
import time                                          # For adding delays
import cv2                                           # For capturing video and basic image processing
from picarx import Picarx                            # For controlling the PicarX robot car


# -------------------------
# INITIAL SETUP
# -------------------------
px = Picarx()  # Initialize the PicarX car


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def _read_label_file(file_path):
    """
    Read the labels.txt file (provided by Coral).
    This file maps each class ID to its corresponding object name.

    Args:
        file_path (str): The path to the labels.txt file.

    Returns:
        dict: A dictionary that maps class IDs (int) to object names (str).
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        # pair[0] is the class ID (as a string), pair[1] is the object name
        ret[int(pair[0])] = pair[1].strip()

    return ret


def move_car(direction):
    """
    Move or steer the car based on the ball's detected position.

    Args:
        direction (str): The direction of the ball, one of:
                         "top", "left", "right", or "center"/other.
    """
    if direction == "top":
        # Go straight ahead if the ball is at the top (near the top edge of the frame)
        px.set_dir_servo_angle(0)  # Straight steering
        px.forward(2)             # Move forward at speed 2
        print(f"Direction {direction}. Moving forward")

    elif direction == "left":
        # Turn left if the ball is near the left side of the frame
        px.set_dir_servo_angle(-30)  # Slightly left steering
        px.forward(2)               # Move forward at speed 2
        print(f"Direction {direction}. Moving left")

    elif direction == "right":
        # Turn right if the ball is near the right side of the frame
        px.set_dir_servo_angle(30)  # Slightly right steering
        px.forward(2)              # Move forward at speed 2
        print(f"Direction {direction}. Moving right")

    else:
        # If ball is "center" or any other state, we do not move the car
        print(f"Direction {direction}. No action")


def locate_object(top_x, top_y, image_width, image_height, margin=0.4):
    """
    Determine the rough position of an object in the image
    (e.g., "left", "right", "top", "bottom", or "center").

    Args:
        top_x (int): X-coordinate of the object's top-left corner.
        top_y (int): Y-coordinate of the object's top-left corner.
        image_width (int): The width of the current frame.
        image_height (int): The height of the current frame.
        margin (float): The margin around the edges that defines "around"
                        (default is 0.4, meaning 40% of the edges).

    Returns:
        str: Position of the object relative to the image:
             - "left"   if it's in the left region
             - "right"  if it's in the right region
             - "top"    if it's in the top region
             - "bottom" if it's in the bottom region
             - "center" otherwise
    """
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


# -------------------------
# MAIN FUNCTION
# -------------------------
def main():
    """
    Main function that:
    1. Initializes the camera and Edge TPU model.
    2. Continuously captures frames from the camera.
    3. Detects a "sports ball" in each frame.
    4. Directs the PicarX car based on where the ball is found.
    5. Stops the car if the ball is in the center of the image.
    """
    # -------------------------
    # MODEL & LABELS SETUP
    # -------------------------
    model_filename = "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    label_filename = "coco_labels.txt"

    # Create interpreter to run our model on the Edge TPU
    interpreter = make_interpreter(model_filename)
    interpreter.allocate_tensors()

    # Read the label file
    labels = _read_label_file(label_filename)

    # -------------------------
    # CAMERA SETUP
    # -------------------------
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    # Using GStreamer pipeline for Raspberry Pi camera
    gst_pipeline = (
        f"libcamerasrc ! "
        f"video/x-raw, width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, framerate=10/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink"
    )

    # Initialize video capture using the GStreamer pipeline
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open GStreamer pipeline")
        return

    # Process every frame (frame_skip=1 means no skipping)
    frame_skip = 1
    frame_counter = 0

    # -------------------------
    # FRAME CAPTURE LOOP
    # -------------------------
    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Keep track of the current frame index
            frame_counter += 1

            # If we want to skip frames, we'd check here.
            if frame_counter % frame_skip != 0:
                continue

            # -------------------------
            # PREPROCESSING FOR MODEL
            # -------------------------
            input_shape = interpreter.get_input_details()[0]['shape']
            resized_image = cv2.resize(frame, (input_shape[1], input_shape[2]))
            input_data = resized_image.astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

            # Set the input tensor for the model
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)

            # Run the detection model
            interpreter.invoke()

            # -------------------------
            # MODEL OUTPUT
            # -------------------------
            # Extract detection results from the model's output tensors
            boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])
            scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])
            num_detections = int(interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0])

            position = None

            # Loop through each detected object
            for i in range(num_detections):
                class_id = int(classes[0][i])     # ID of the detected class
                score = scores[0][i]             # Confidence score of the detection

                # class_id == 36 usually corresponds to "sports ball" in COCO dataset
                # We only proceed if we are fairly confident (score > 0.05)
                if class_id == 36 and score > 0.05:
                    h, w = frame.shape[:2]       # height and width of the camera frame

                    # The coordinates are normalized (between 0 and 1),
                    # so we multiply by frame dimensions to get actual pixel values.
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    left = int(xmin * w)
                    bottom = int(ymax * h)

                    # Determine where the ball is located in the frame
                    position = locate_object(left, bottom, w, h)
                    print(f"The object is around the {position} of the image.")

                    # Tell the car how to move based on the ball's position
                    if position != "center":
                        move_car(position)
                    else:
                        # Stop the car if the ball is near the center
                        px.forward(0)

            # -------------------------
            # POST-INFERENCE ACTIONS
            # -------------------------
            # Sleep a bit to avoid maxing out the CPU
            time.sleep(0.1)

            # After each loop, reset steering and stop the car
            px.set_dir_servo_angle(0)
            px.forward(0)

    finally:
        # -------------------------
        # CLEANUP
        # -------------------------
        # Release the camera and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


# -------------------------
# RUN THE MAIN FUNCTION
# -------------------------
if __name__ == "__main__":
    main()
