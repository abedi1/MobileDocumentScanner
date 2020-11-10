#imports
import numpy as np
import cv2

def order_points(pts):
    # initialize a list of coordinates that will be orderd
    # such that the entry order in the list is the top-left,
    # top-right, bottom-right and bottom-left
    # creates an array filled with zeroes where each entry is a tuple
    rect = np.zeros((4,2), dtype = "float32")

    #the top-left point wil have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1) #sum of the x and y cordinate of each cell
    rect[0] = pts[np.argmin(s)] #basic fact of coordinates
    rect[2] = pts[np.argmax(s)]

    #now, compute the difference between the points,
    # the top-right point will ahve the smallest difference
    # the bottom left will have largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    #return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    #obtain a consistent order of the points and unpack them
    #individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect # getting them out of the array

    #compute the width of hte new image
    #max distance between br and bl x cordinates
    # and tr and tl x cordinates (do pythagoras to account for potential shift in image)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    #compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Gets the perspective transformation matrix from og points (rect) to new points (dst)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth,maxHeight)) #applies matrix to image

    #return the warped image
    return warped