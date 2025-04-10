from picamera2 import Picamera2
import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

# Setup GPIO
light_pin = 17  # Adjust as per your setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(light_pin, GPIO.OUT)

# Initialize Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()

# Allow the camera to warm up
time.sleep(0.1)

# Define color ranges in HSV for green
color_ranges = {
    "Green": [(50, 100, 50), (70, 255, 255)]  # HSV range for green
}

try:
    while True:
        # Capture an image to a numpy array
        buffer = picam2.capture_array("main")
        image = np.array(buffer)

        # Convert from RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        green_detected = False

        # Check for green color
        for color_name, (lower, upper) in color_ranges.items():
            lower_np = np.array(lower)
            upper_np = np.array(upper)

            # Create a mask for green color
            mask = cv2.inRange(hsv_image, lower_np, upper_np)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if any contours are detected and their area
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    green_detected = True
                    break  # Stop checking once green is detected

        # Control the light based on color detection
        GPIO.output(light_pin, green_detected)

        # Display the image in its original RGB format
        cv2.imshow("Frame", image)  # No conversion needed for display

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    GPIO.cleanup()
    picam2.stop()
