import cv2
import numpy as np
from sklearn.cluster import KMeans

font = cv2.FONT_HERSHEY_SIMPLEX

class Tracker:
    def __init__(self):
        # What mode?
        self.mode = 'learn'
        # Properties about the learnt object
        self.typical_rgb = np.array([0, 0, 0])
        self.max_rgb = np.array([0, 0, 0])
        self.min_rgb = np.array([0, 0, 0])
        # Properties about the tracked object
        self.objects = {}
        self.location = [0, 0]
        self.size = [0, 0]
        self.angle = 0
        self.last_seen = 0
        
    def new_image(self, image):
        if self.mode == 'learn':
            self._learn_block(image)
        elif self.mode == 'tracker':
            return self._get_block(image)

    def _learn_block(image):
        pass

    def _get_block(self, image):
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
                im2 = cv2.rectangle(image,(x, y), (x+w, y+h),(255,0,0),3)
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

    
class Learn:
    def __init__(self):
        # Image to extract colour values from
        self.image = ""
        
    def make_histogram(self, cluster):
        #Count the number of pixels in each cluster
        #:param: KMeans cluster
        #:return: numpy histogram
        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        hist = hist.astype('float32')
        hist /= hist.sum()
        return hist
    
    def make_bar(self, height,
                 width, colour):
        #Create an image of a given colour
        #:param: height of the image
        ##:param: width of the image
        #:param: BGR pixel values of the colour
        #:return: tuple of bar, rgb values, and hsv values
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = colour
        red, green, blue = int(colour[2]), int(colour[1]), int(colour[0])
        hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_bar[0][0]
        return bar, (red, green, blue), (hue, sat, val)
    
    def get_dominant_colour(self):
        img = cv2.imread(self.image)
        height, width, _ = np.shape(img)
        # reshape the image to be a simple list of RGB pixels
        image = img.reshape((height * width, 3))
        # pick the most common colours, in this case [1]
        num_clusters = 1 
        clusters = KMeans(n_clusters=num_clusters)
        clusters.fit(image)
        # count the dominant colors and put them in "buckets"
        histogram = self.make_histogram(clusters)
        # then sort them, most-common first
        combined = zip(histogram, clusters.cluster_centers_)
        combined = sorted(combined, key=lambda x: x[0], reverse=True)
        bars = []
        hsv_values = []
        for index, rows in enumerate(combined):
            bar, rgb, hsv = self.make_bar(100, 100, rows[1])
            hsv_values.append(hsv)
            bars.append(bar)
        bgr = reversed(rgb)
        print(tuple(bgr))
        # Outputs the colour in BGR format 
        return tuple(bgr)
