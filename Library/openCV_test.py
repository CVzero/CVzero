import numpy as np
import cv2

# Setup video capture 
cap = cv2.VideoCapture(0)
# Reset frame count
counter = 0
# Set font for text
font = cv2.FONT_HERSHEY_SIMPLEX
while True:

    _, img = cap.read()

    cv2.resize(img, (600, 480)) 
    # Convert image type to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Add some blur to reduce noise
    blur = cv2.GaussianBlur(img, (11, 11), 0)

    # Define colour hue values to look for
    blue_lower = np.array([110,50,50])
    blue_upper = np.array([130,255,255])

    # Look for colours in that hue range
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    # Get rid of some noise
    mask = cv2.erode(blue, None, iterations=2)
    
    # Find the contours of the blob so we can draw a rectangle around it
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
    # Draw the rectangle
    for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                        
                    x,y,w,h = cv2.boundingRect(contour)     
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                    # Coordinates??
                    cv2.putText(img, "Object at: x: {} y: {} w: {} h: {}".format(x, y, w, h),(300,100), font, 0.5,(255,255,255),2)  

    # Frame counter
    counter += 1
                        
    cv2.putText(img, "Frame: {} ".format(counter),(20,100), font, 0.5,(255,255,255),2)  


    # Show the result
    cv2.imshow("Tracking", img)

    
    # Quitting code
    if cv2.waitKey(10) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                break

    

