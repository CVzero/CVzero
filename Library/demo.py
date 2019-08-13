import cv2, numpy as np, time
import picamera.array
import picamera
import cvzero

# Setup Pi Camera
camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCap = picamera.array.PiRGBArray(camera, size=(640, 480))

# Wait for camera to warm-up
time.sleep(0.1)

# Setup CVzero tracker
tracker = CVzero.Tracker()
tracker.mode = "tracker"
tracker.min_rgb = np.array([105,50,50])
tracker.max_rgb = np.array([130,255,255])

# Frame count
frames = 0
# Start time for FPS calculation
t = time.time()

for frame in camera.capture_continuous(rawCap, format="bgr", use_video_port=True):
    # Track objects using CVzero tracker
    im2, objects_detected = tracker.new_image(frame.array)
    # Calculate and print FPS
    # print("FPS:", frames/(time.time() - t))
    # Show tracker output
    cv2.imshow("out", im2)
    # Print parameters of each object
    for obj in objects_detected.keys():
        [x, y, w, h] = objects_detected[obj]
        print(obj,":\t", "x:", x, "Y:", y, "W:", w, "H:", h)
    
    # increment frame count
    frames += 1

    key = cv2.waitKey(1) & 0xFF
    rawCap.truncate(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()        