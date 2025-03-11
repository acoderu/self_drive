import cv2
import numpy as np

# ----------------------------------------------------------------------
# You must fill in these values with the results of your camera calibration!
# For a fisheye lens, ideally you use cv2.fisheye.calibrate(), but some
# people still use pinhole-based calibrations with cv2.undistort() if the distortion is mild.
# ----------------------------------------------------------------------
camera_matrix = np.array([
    [250.0,   0.0, 160.0],  # Focal lengths & principal point X
    [  0.0, 250.0, 160.0],  # Focal lengths & principal point Y
    [  0.0,   0.0,   1.0 ]
])
dist_coeffs = np.array([-0.02, 0.01, 0.0, 0.0])  # Example placeholder values
# ----------------------------------------------------------------------


def detect_yellow_ball(frame):
    """
    Detect a yellow tennis ball in a given BGR frame and return:
      - The bounding box (x, y, w, h)
      - The mask used to find the ball (for visualization)
      - The contours found in that mask
    Returns None, None, None if no ball is detected.
    
    Parameters:
    -----------
    frame : np.ndarray
        The undistorted image in BGR color format.
    
    Returns:
    --------
    best_box : tuple or None
        (x, y, w, h) bounding box of the detected tennis ball.
    mask : np.ndarray
        The binary mask that highlights the yellow color.
    contours : list
        List of all contours found in the mask.
    """
    
    # 1) Convert from BGR to HSV color space.
    #    HSV makes it easier to isolate specific colors (like "tennis ball" yellow).
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) Define the lower and upper HSV bounds for "tennis ball yellow."
    #    These ranges may need to be tweaked depending on your lighting.
    lower_yellow = np.array([16, 40, 0], dtype=np.uint8)
    upper_yellow = np.array([99, 255, 255], dtype=np.uint8)

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
        
        print (f"got area {area}. got cicularity {circularity}")
        # Filter by area (not too small/large) AND by circularity (close to 1 means "circle").
        if 120 < area < 700 and 0.7 < circularity < 1.2:
            # If it passes the test, get the bounding box (rectangle) for the contour
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check if this contour is better (larger radius) than our current best
            if radius > best_radius:
                best_radius = radius
                best_box = (x, y, w, h)

    return best_box, mask, contours

def undistort_fisheye(image):
        """
        Removes (approx) fisheye distortion from the camera.
        Then crops the edges to reduce artifacts.
        """
        h, w = image.shape[:2]

        # K is the camera matrix, D is distortion params
        K = np.array([[w / 1.5, 0, w / 2],
                      [0, w / 1.5, h / 2],
                      [0, 0, 1]],
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
        cropped = undistorted[int(h * 0.1):int(h * 0.9),
                  int(w * 0.1):int(w * 0.9)]

        return cropped

def main():
    """
    Main loop:
    - Opens a camera feed (320x320 if supported).
    - Corrects for fisheye distortion.
    - Detects the yellow tennis ball.
    - Draws a bounding box around it on the original frame.
    - Also displays a second window with the black background + white contours for visualization.
    - Quits when 'q' is pressed.
    """

    # 1) Open a connection to the camera (0 is typically your first camera).
    cap = cv2.VideoCapture(0)

    # 2) Optionally set the camera resolution to 320x320 if supported.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    
    while True:
        # 3) Capture a frame from the camera.
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from the camera.")
            break

        # 4) Undistort the frame using our camera matrix and distortion coefficients.
        #    For fisheye lenses, if you used cv2.fisheye.calibrate(), you might need
        #    cv2.fisheye.undistortImage() or a slightly different approach.
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        #undistorted_frame = undistort_fisheye(frame)

        # 5) Detect the yellow ball in the undistorted image.
        #best_box, mask, contours = detect_yellow_ball(undistorted_frame)
        best_box, mask, contours = detect_yellow_ball(undistorted_frame)

        # 6) If the tennis ball was found, draw a bounding box around it on the original frame.
        if best_box is not None:
            x, y, w, h = best_box
            cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(undistorted_frame, "Tennis Ball Detected", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7) Create a blank (black) image to draw contours on, so we can see them clearly.
        #    We make it the same size as the mask (320x320 if camera is at that resolution).
        contour_visual = np.zeros_like(mask)  # black & white image (1 channel if mask is 1 channel)

        # Draw all contours in white color (value=255).
        # thickness=1 means a thin line around each shape.
        cv2.drawContours(contour_visual, contours, -1, (255), 1)

        # 8) Display both the detection window and the contour visualization window.
        cv2.imshow("Undistorted Frame with Detection", undistorted_frame)
        cv2.imshow("Mask Contours (Black & White)", contour_visual)

        # 9) Check if the user pressed 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 10) Release the camera resource and close all windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
