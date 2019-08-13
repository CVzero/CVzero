import cv2
import numpy as np

blue_lower = np.array([110,50,50])
blue_upper = np.array([130,255,255])




class get:

    
    def __init__(self, source):

        self.counter = 0

        self.cap = cv2.VideoCapture(source)
        

        

        

    def blocks_between_hue(self, lower_value, upper_value):
         _, img = self.cap.read()
         # Convert image type to HSV
         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
         # Add some blur to reduce noise
        # blur = cv2.GaussianBlur(hsv, (11, 11), 0)
         # Look for colours in that hue range
         block = cv2.inRange(hsv, lower_value, upper_value)
         # Get rid of some noise
         mask = cv2.erode(block, None, iterations=2)
         # Find the contours of the blob so we can draw a rectangle around it
         contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

         for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                    x,y,w,h = cv2.boundingRect(contour)     
                    im2 = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                    return(x,y,w,h)
         
        
        
get = get(0)
while True:
    detect_block = get.blocks_between_hue(blue_lower, blue_upper)
    print(detect_block)

