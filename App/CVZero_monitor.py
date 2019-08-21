import cv2
import numpy as np
import picamera
import picamera.array
import time
from sklearn.cluster import KMeans

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
    cv2.putText(temp, "Select Object and press enter to learn it, or 'C' to Cancel", (10, 20), font, 0.75, (0, 100, 255), 1, cv2.LINE_AA)
    # Get start and end coordinate pairs of user drawn ROI rectangle
    selection = cv2.selectROI("CV Zero Monitor", temp, False, False)
    # return ROI of original image using coordiantes in "selection"
    return image[selection[1]:selection[1]+selection[3], selection[0]:selection[0]+selection[2]]


def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar


def k_means(image, numOfClusters):
    colours = []
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters = numOfClusters)
    clt.fit(image)

    # build a histogram of clusters
    hist = centroid_histogram(clt).tolist()
    hist = [hist, []]
    # Sort colour cluster by their % abundance in the object ROI
    for (dominance, colour) in sorted(zip(hist[0], clt.cluster_centers_)):
        hist[1].append(dominance)
        colours.append(colour.astype("uint8"))
    
    # Round % abundance of each colour to 1 D.P
    hist = [_ for _ in map(lambda x: round(x, 2) , hist[1])]
    # print("hist: ", hist)

    bar = plot_colors(hist, colours)
    cv2.imshow("bar", bar)
    # Filter out colour that have a % abundance below the threshold varaiable "thresh"
    for (dominance, colour) in zip(hist.copy(), colours.copy()):
        if dominance < thresh:
            del hist[0]
            del colours[0]
        else:
            print(dominance, colour)

    bar2 = plot_colors(hist, colours)
    cv2.imshow("bar after thresh", bar2)

    return np.amin(colours, axis=0), np.amax(colours, axis=0)


# Set a fixed 5% threshold (could be made user modifiable using sliders in the next commit)
thresh = 0.05

for frame in camera.capture_continuous(rawCap, format="bgr", use_video_port=True):
    # copy image to overlay guide text
    out = frame.array.copy()
    # Add guide text to temp image
    cv2.putText(out, "Press 'L' to enter Learn Mode", (10, 20), font, 0.75, (0, 100, 255), 1, cv2.LINE_AA)
    # Show frame with guide text
    cv2.imshow("CV Zero Monitor", out)

    key = cv2.waitKey(1) & 0xFF
    # Clear stream
    rawCap.truncate(0)

    # Activate Learn mode if 'L' is pressed
    if key == ord("l"):
        roi = select_roi(frame.array)
        if len(roi) > 0:
            cv2.imshow("roi", roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            cv2.imshow("image HSV", roi)
            HSV_LB, HSV_UB = k_means(roi, 5)
            BGR_LB, BGR_UB = cv2.cvtColor(np.array([[HSV_LB]]), cv2.COLOR_HSV2BGR), cv2.cvtColor(np.array([[HSV_UB]]), cv2.COLOR_HSV2BGR)
            print("HSV_LB:", HSV_LB, "HSV_UB:", HSV_UB, "BGR_LB:", BGR_LB, "BGR_UB:", BGR_UB)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
