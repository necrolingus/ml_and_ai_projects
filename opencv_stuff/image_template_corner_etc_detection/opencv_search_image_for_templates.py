#https://pythonprogramming.net/template-matching-python-opencv-tutorial/?completed=/canny-edge-detection-gradients-python-opencv-tutorial/

import cv2
import numpy as np

img_rgb = cv2.imread('rasppis.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


#What this ".shape" thing mean further down below
# Your image shape returns 3 dimensions

# im.shape
# >>> (24, 28, 3)
# If you only want the first 2 do:

# w, h = im.shape[:-1]
# >>> (24, 28)
# or

# w, h, _ = im.shape
# w is 24, h is 28
# The _ is like a convention in Python for variables you don't want to use, or a "throwaway".


template = cv2.imread('rasppis_template.jpg', 0) #0 means gray

# We load the template and note the dimensions.
w, h = template.shape[::-1]
#print(w, h) #will print 19, 22. I.e. the image size



#Here, we call res the matchTemplate between the img_gray (our main image), the template, and then the matching method we're 
#going to use. We specify a threshold, here 0.8 for 80%. Then we find locations with a logical statement, where the res is 
#greater than or equal to 80%.

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)


#Finally, we mark all matches on the original image, using the coordinates we found in the gray image:
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


#extracting foreground
#https://pythonprogramming.net/grabcut-foreground-extraction-python-opencv-tutorial/?completed=/template-matching-python-opencv-tutorial/







