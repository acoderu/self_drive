import cv2
import numpy as np

def nothing(x):
    """
    This function does nothing and is just a placeholder for the trackbar callbacks.
    OpenCV trackbars require a callback function even if it does nothing.
    """
    pass

def main():
    # 1) Open a connection to your camera.
    #    Change the index if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    # 2) Optionally set resolution to 320x320 (if supported).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    # 3) Create a window to hold the trackbars.
    cv2.namedWindow("Trackbars")

    # 4) Create 6 trackbars for Lower and Upper HSV bounds.
    #    The maximum Hue in OpenCV is 179, while Sat and Val max out at 255.
    cv2.createTrackbar("LowerH", "Trackbars", 20, 179, nothing)
    cv2.createTrackbar("LowerS", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("LowerV", "Trackbars", 100, 255, nothing)

    cv2.createTrackbar("UpperH", "Trackbars", 30, 179, nothing)
    cv2.createTrackbar("UpperS", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("UpperV", "Trackbars", 255, 255, nothing)

    while True:
        # 5) Read a frame from the camera.
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # ------------------------------------------------------------
        # (If using a fisheye lens, insert your undistort logic here!)
        # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        # For demonstration, let's just use the raw frame:
        # ------------------------------------------------------------
        undistorted_frame = frame

        # 6) Convert from BGR to HSV.
        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)

        # 7) Read the trackbar positions (i.e. the current slider values).
        lowerH = cv2.getTrackbarPos("LowerH", "Trackbars")
        lowerS = cv2.getTrackbarPos("LowerS", "Trackbars")
        lowerV = cv2.getTrackbarPos("LowerV", "Trackbars")

        upperH = cv2.getTrackbarPos("UpperH", "Trackbars")
        upperS = cv2.getTrackbarPos("UpperS", "Trackbars")
        upperV = cv2.getTrackbarPos("UpperV", "Trackbars")

        # 8) Create the NumPy arrays for the lower and upper bounds.
        lower_bound = np.array([lowerH, lowerS, lowerV], dtype=np.uint8)
        upper_bound = np.array([upperH, upperS, upperV], dtype=np.uint8)

        # 9) Create a mask using the current HSV range.
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 10) Optional: You can run morphological operations to clean up noise in the mask.
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        # 11) Show the original (undistorted) frame in one window.
        cv2.imshow("Frame", undistorted_frame)

        # 12) Show the mask in another window.
        #     White pixels in the mask represent the color range we currently have set.
        cv2.imshow("Mask", mask)

        # 13) If you want to see bounding boxes for the ball, you could do:
        #     - Find contours in the mask
        #     - Filter by area or shape
        #     - Draw bounding boxes on "undistorted_frame"
        # For now, we are just focusing on picking good HSV ranges with the trackbars.

        # 14) Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 15) Release the camera and close the windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
