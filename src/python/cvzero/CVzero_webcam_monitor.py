from tkinter import *
from PIL import ImageTk, Image
import cv2
from sklearn.cluster import KMeans
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.pack()
# Create a label in the frame
lmain = Label(app)
lmain.pack()

# Capture from camera
cap = cv2.VideoCapture(0)

def select_roi(image):
    # copy image temporarily to overlay guide text
    temp = image.copy()
    # Add guide text to temp image
    cv2.putText(temp, "Select Object and press enter to learn it, or 'C' to Cancel", (10, 20), font, 0.5, (0, 100, 255), 1, cv2.LINE_AA)
    # Get start and end coordinate pairs of user drawn ROI rectangle
    selection = cv2.selectROI("Select object", temp, False, False)
    # return ROI of original image using coordiantes in "selection"
    cv2.destroyAllWindows()
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

# Run this when the learn object button is pressed and start the k means sorting
def learn_button_handler():
    global out
    roi = select_roi(out)
    if len(roi) > 0:
        cv2.imshow("roi", roi)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cv2.imshow("image HSV", roi)
        HSV_LB, HSV_UB = k_means(roi, 1)
        BGR_LB, BGR_UB = cv2.cvtColor(np.array([[HSV_LB]]), cv2.COLOR_HSV2BGR), cv2.cvtColor(np.array([[HSV_UB]]), cv2.COLOR_HSV2BGR)
        print("HSV_LB:", HSV_LB, "HSV_UB:", HSV_UB, "BGR_LB:", BGR_LB, "BGR_UB:", BGR_UB)
    
# function for video streaming into tkinter window
def video_stream():
    _, frame = cap.read()
    global out
    out = frame.copy()   
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)

# Setup tkinter widgets
menubar = Menu(root)
root.config(menu=menubar)
fileMenu = Menu(menubar)
helpMenu = Menu(menubar)
aboutMenu = Menu(menubar)
fileMenu.add_command(label="Exit", command= lambda : exit())
menubar.add_cascade(label="File", menu=fileMenu)
menubar.add_cascade(label="Help", menu=helpMenu)
menubar.add_cascade(label="About", menu=aboutMenu)

b = Button(root, height=2, text="Learn object", command = lambda : learn_button_handler())
b.pack(side=LEFT)
b = Button(root, height=2, width=10, text="Quit", command = lambda : exit())
b.pack(side=LEFT)
w = Scale(root, from_=0, to=0.1,resolution=0.01, orient=HORIZONTAL)
# Set the default threshold to 5%
w.set(0.05)
w.pack(side=RIGHT)
l = Label(root, text="Colour Threshold:")
l.pack(side=RIGHT)

# Get slider value and set that as the colour threshold
thresh = int(w.get())

root.title("CVzero Monitor tool")
root.resizable(0,0)
video_stream()
root.mainloop()
