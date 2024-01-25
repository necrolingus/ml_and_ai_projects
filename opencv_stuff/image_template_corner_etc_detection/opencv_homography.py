#https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/?completed=/corner-detection-python-opencv-tutorial/

#Feature matching is going to be a slightly more impressive version of template matching, 
#where a perfect, or very close to perfect, match is required.

#We start with the image that we're hoping to find, and then we can search for this image within another image. 
#The beauty here is that the image does not need to be the same lighting, angle, rotation...etc. The features 
#just need to match up.


import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv-feature-matching-template.jpg',0)
img2 = cv2.imread('opencv-feature-matching-image.jpg',0)

#This is the detector we're going to use for the features.
orb = cv2.ORB_create() #ORB is oriented fast and rotated brief.

#Here, we find the key points and their descriptors with the orb detector.
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with 
#all other features in second set using some distance calculation. And the closest one is returned.
#For binary string based descriptors like ORB, BRIEF, BRISK etc, cv2.NORM_HAMMING should be used.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Here we create matches of the descriptors, then we sort them based on their distances.
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance) #we sort to get the best matches, otherwise we match all over the show. Many false positives


img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()