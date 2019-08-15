import cv2, numpy as np, picamera, picamera.array, time

# Setup Pi Camera
camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCap = picamera.array.PiRGBArray(camera, size=(640, 480))

# Wait for camera to warm-up
time.sleep(0.1)

font = cv2.FONT_HERSHEY_SIMPLEX

def select_roi(image):
    # copy image temporarily to overlay guide text
    temp = image.copy()
    # Add guide text to temp image
    cv2.putText(temp, "Select Object and press enter to learn it, or 'C' to Cancel", (10, 20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # Get start and end coordinate pairs of user drawn ROI rectangle
    selection = cv2.selectROI("CV Zero Monitor", temp, False, False)
    # return ROI of original image using coordiantes in "selection"
    return image[selection[1]:selection[1]+selection[3], selection[0]:selection[0]+selection[2]]

for frame in camera.capture_continuous(rawCap, format="bgr", use_video_port=True):
    # copy image temporarily to overlay guide text
    temp = frame.array.copy()
    # Add guide text to temp image
    cv2.putText(temp, "Press 'L' to enter Learn Mode", (10, 20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # Show frame with guide text
    cv2.imshow("CV Zero Monitor", temp)    
    
    key = cv2.waitKey(1) & 0xFF
    # Clear stream
    rawCap.truncate(0)

    # Activate Learn mode if 'L' is pressed
    if key == ord("l"):
        roi = select_roi(frame.array)
        if len(roi) > 0:
            cv2.imshow("roi", roi)
    if key == ord("q"):
        break

cv2.destroyAllWindows()