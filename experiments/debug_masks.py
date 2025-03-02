#!/usr/bin/env python3  

import cv2  
import numpy as np  
from picamera2 import Picamera2  
import time

class HSVCalibrator:  
    def __init__(self):  
        # Initialize Picamera2  
        self.picam2 = Picamera2()  
        config = self.picam2.create_preview_configuration(  
            main={"size": (1280, 720)}  
        )    
        self.picam2.configure(config)  
        
        self.picam2.start()  
        time.sleep(2)  # Warm-up time  

        # Create windows  
        cv2.namedWindow('Original')  
        cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)  
        cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)  

        # Create trackbars  
        self.create_trackbars()  

    def create_trackbars(self):  
        """  
        Create trackbars for HSV range adjustment  
        Multiple methods to isolate color  
        """  
        cv2.createTrackbar('Method', 'Trackbars', 0, 4, self.nothing)  
        
        # HSV Range Trackbars  
        cv2.createTrackbar('LH', 'Trackbars', 0, 179, self.nothing)  
        cv2.createTrackbar('LS', 'Trackbars', 0, 255, self.nothing)  
        cv2.createTrackbar('LV', 'Trackbars', 0, 255, self.nothing)  
        cv2.createTrackbar('UH', 'Trackbars', 179, 179, self.nothing)  
        cv2.createTrackbar('US', 'Trackbars', 255, 255, self.nothing)  
        cv2.createTrackbar('UV', 'Trackbars', 255, 255, self.nothing)  

        # Additional processing trackbars  
        cv2.createTrackbar('Blur', 'Trackbars', 0, 10, self.nothing)  
        cv2.createTrackbar('Erode', 'Trackbars', 0, 10, self.nothing)  
        cv2.createTrackbar('Dilate', 'Trackbars', 0, 10, self.nothing)  

    def nothing(self, x):  
        """Placeholder callback for trackbars"""  
        pass  

    def process_frame(self):  
        """  
        Capture and process frame with multiple color isolation techniques  
        """  
        # Capture frame  
        frame = self.picam2.capture_array()  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

        # Get trackbar values  
        method = cv2.getTrackbarPos('Method', 'Trackbars')  
        
        # HSV Range  
        lh = cv2.getTrackbarPos('LH', 'Trackbars')  
        ls = cv2.getTrackbarPos('LS', 'Trackbars')  
        lv = cv2.getTrackbarPos('LV', 'Trackbars')  
        uh = cv2.getTrackbarPos('UH', 'Trackbars')  
        us = cv2.getTrackbarPos('US', 'Trackbars')  
        uv = cv2.getTrackbarPos('UV', 'Trackbars')  

        # Processing parameters  
        blur = cv2.getTrackbarPos('Blur', 'Trackbars')  
        erode_iter = cv2.getTrackbarPos('Erode', 'Trackbars')  
        dilate_iter = cv2.getTrackbarPos('Dilate', 'Trackbars')  

        # Convert to HSV  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  

        # Define HSV range  
        lower = np.array([lh, ls, lv])  
        upper = np.array([uh, us, uv])  

        # Color isolation methods  
        if method == 0:  # Standard HSV Threshold  
            mask = cv2.inRange(hsv, lower, upper)  
        
        elif method == 1:  # Gaussian Blur + Threshold  
            blurred = cv2.GaussianBlur(hsv, (5, 5), blur+1 if blur > 0 else 1)  
            mask = cv2.inRange(blurred, lower, upper)  
        
        elif method == 2:  # Adaptive Thresholding  
            gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  
            mask = cv2.adaptiveThreshold(  
                gray, 255,   
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   
                cv2.THRESH_BINARY_INV,   
                11, 2  
            )  
        
        elif method == 3:  # Otsu's Thresholding  
            gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  
            _, mask = cv2.threshold(  
                gray, 0, 255,   
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  
            )  
        
        elif method == 4:  # Color Channel Separation  
            h, s, v = cv2.split(hsv)  
            mask = cv2.inRange(h, lh, uh)  

        # Optional Morphological Operations  
        if erode_iter > 0:  
            kernel = np.ones((3,3), np.uint8)  
            mask = cv2.erode(mask, kernel, iterations=erode_iter)  
        
        if dilate_iter > 0:  
            kernel = np.ones((3,3), np.uint8)  
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)  

        # Display results  
        cv2.imshow('Original', frame)  
        cv2.imshow('HSV', mask)  

        return mask  

    def run(self):  
        """  
        Main run method with continuous processing  
        """  
        try:  
            while True:  
                # Process frame  
                self.process_frame()  

                # Exit on 'q' key  
                key = cv2.waitKey(1) & 0xFF  
                if key == ord('q'):  
                    break  

        except KeyboardInterrupt:  
            print("Calibration stopped.")  
        
        finally:  
            # Cleanup  
            self.picam2.stop()  
            cv2.destroyAllWindows()  

def main():  
    calibrator = HSVCalibrator()  
    calibrator.run()  

if __name__ == "__main__":  
    main()