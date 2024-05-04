#https://pythonprogramming.net/loading-images-python-opencv-tutorial/
#at some point there is no link to the next tutorial, I think this is the next one https://pythonprogramming.net/image-operations-python-opencv-tutorial/



import cv2
import matplotlib
import numpy
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE) #IMREAD_UNCHANGED
#or the second parameter, you can use -1, 0, or 1. Color is 1, grayscale is 0, 
#and the unchanged is -1. Thus, for grayscale, one could do img = cv2.imread('watch.jpg', 0)


#If you do not have a webcam, 
#this will be the main method you will use throughout this tutorial, loading an image.

cv2.imshow('image', img)
cv2.waitKey(0)
#o wait until any key is pressed. Once that's done, we use 
#cv2.destroyAllWindows() to close everything.
cv2.destroyAllWindows()
cv2.imwrite('watchgray.png', img)

#showing an image with matplotlib
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
# plt.show()



#Using video feeds
cap = cv2.VideoCapture(0) #This will return video from the first webcam on your computer.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


#This code initiates an infinite loop (to be broken later by a break statement), 
#where we have ret and frame being defined as the cap.read(). Basically, ret 
#is a boolean regarding whether or not there was a return at all, and the frame 
#is each frame that is returned. If there is no frame, you wont get an error, you will get None

while(True):
	ret, frame = cap.read()
	
	#Here, we define a new variable, gray, as the frame, converted to gray. Notice this says BGR2GRAY. 
	#It is important to note that OpenCV reads colors as BGR (Blue Green Red), where most computer applications 
	#read as RGB (Red Green Blue). Remember this.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	out.write(frame)
	
	#Notice that, despite being a video stream, we still use imshow. Here, we're showing the converted-to-gray 
	#feed. If you wish to show both at the same time, you can do imshow for the original frame, and imshow 
	#for the gray and two windows will appear.
	cv2.imshow('frame', gray)
	
	#This statement just runs once per frame. Basically, if we get a key, and that key is a q, 
	#we will exit the while loop with a break, which then the release and destroy code
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
		
cap.release()
out.release()
cv2.destroyAllWindows()
















