#imports
from transform.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

#construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

#load the image and compute the ratio
# between old and new height
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0 #ratio between two images
orig = image.copy()
image = imutils.resize(image, height = 500) #resize image to 500 pixels

#convert the image to grayscale, blur it
# find edges in the images with canny
gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0) #helps remove noise
edged = cv2.Canny(gray, 75, 200)

#show the original image and the edge detected image
print("Step 1: Edge detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find the contours in the edged image,
#assuming the largest ones are our edges and initalize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted (cnts, key = cv2.contourArea, reverse = True) [:5]

# loop over the contours
for c in cnts:
	#approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has 4 points then we,
	# can assume that we have found our document
	if len (approx) == 4:
		screenCnt = approx
		break

# show the contour (outline) of the paper
print("Step 2: find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0) , 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio) #resize it back

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect (optional)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

#how the original and scanned images
print("Step 3: apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
