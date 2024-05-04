#https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/?completed=/corner-detection-python-opencv-tutorial/

#Feature matching is going to be a slightly more impressive version of template matching, 
#where a perfect, or very close to perfect, match is required.
#We start with the image that we're hoping to find, and then we can search for this image within another image. 
#The beauty here is that the image does not need to be the same lighting, angle, rotation...etc. The features 
#just need to match up.

import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

#cool tutorial showing how to loop through frames 
#https://www.programcreek.com/python/example/89404/cv2.createBackgroundSubtractorMOG2
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()

#The idea here is to extract the moving forground from the static background. You can also use this to 
#compare two similar images, and immediately extract the differences between them
