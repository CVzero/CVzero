import cv2, numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

class CVzero:
    def __init__(self):
        # Propertieis about the learnt object
        self.typical_rgb = np.array([0, 0, 0])
        self.max_rgb = np.array([0, 0, 0])
        self.min_rgb = np.array([0, 0, 0])
        # Properties about the tracked object
        self.objects = {}
        self.location = [0, 0]
        self.size = [0, 0]
        self.angle = 0
        self.last_seen = 0

    def learnObjects(self, image):
        pass

    def trackObjects(self, image):
        # Flag for whether or not valid objects that pass all filters were detected
        valid_objects = False
        self.objects = {}
        # Convert image type to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Look for colours in that hue range
        block = cv2.inRange(hsv, self.min_rgb, self.max_rgb)
        # Get rid of some noise
        mask = cv2.erode(block, None, iterations=2)
        # Find the contours of the blob so we can draw a rectangle around it
        if int(cv2.__version__[0]) >= 4:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            # Filter by area of contour
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)
                # Draw bounding box     
                im2 = cv2.rectangle(image.copy(),(x,y),(x+w,y+h),(255,0,0),3)
                # Add object ID to center of detected object
                cv2.putText(im2, str(obj_id), (int(x + w/2), int(y + h/2)), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Add object and it's parameters to "objects" dict
                self.objects[obj_id] = [x, y, w, h]
                valid_objects = True
        if valid_objects:
            # return image with overlay if valid objects detected
            return im2, self.objects
        # return input image (no valid objects detected)
        return image, self.objects

