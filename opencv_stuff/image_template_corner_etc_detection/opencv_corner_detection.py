#https://pythonprogramming.net/corner-detection-python-opencv-tutorial/?completed=/grabcut-foreground-extraction-python-opencv-tutorial/
#The purpose of detecting corners is to track things like motion, do 3D modeling, 
#and recognize objects, shapes, and characters.

import cv2
import numpy as np

img = cv2.imread('opencv-corner-detection-sample.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)


#So far, we load the image, convert to gray, then to float32. Next, we detect corners with the goodFeaturesToTrack function. 
#The parameters here are the image, max corners to detect, quality, and minimum distance between corners. As mentioned before, 
#the aliasing issues we have here will allow for many corners to be found, so we put a limit on it.
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
	x,y = corner.ravel()
	cv2.circle(img, (x,y), 3, 255, -1)
	
	
cv2.imshow('Corner', img)
cv2.waitKey(0)
cv2.destroyAllWindows()