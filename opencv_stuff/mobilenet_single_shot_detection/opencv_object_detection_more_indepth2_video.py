#https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/


#In the first part of today’s post on object detection using deep learning we’ll 
#discuss Single Shot Detectors and MobileNets.
#great for resource constrained devices

#When combined together these methods can be used for super fast, real-time object detection on resource 
#constrained devices (including the Raspberry Pi, smartphones, etc.)

#From there we’ll discover how to use OpenCV’s dnn  module to load a pre-trained 
#object detection network. This will enable us to pass input images through the network and 
#obtain the output bounding box (x, y)-coordinates of each object in the image.

#Finally we’ll look at the results of applying the MobileNet Single Shot Detector to example input images.


#When it comes to deep learning-based object detection there are three primary object 
#detection methods that you’ll likely encounter:

#Faster R-CNNs (Girshick et al., 2015)
#You Only Look Once (YOLO) (Redmon and Farhadi, 2015)
#Single Shot Detectors (SSDs) (Liu et al., 2015)




#Faster R-CNNs are likely the most “heard of” method for object detection using deep learning; 
#however, the technique can be difficult to understand (especially for beginners in deep learning), 
#hard to implement, and challenging to train.

#Furthermore, even with the “faster” implementation R-CNNs (where the “R” stands for “Region Proposal”) 
#the algorithm can be quite slow, on the order of 7 FPS.

#If we are looking for pure speed then we tend to use YOLO as this algorithm is much faster, 
#capable of processing 40-90 FPS on a Titan X GPU. The super fast variant of YOLO can even get up to 155 FPS.

#The problem with YOLO is that it leaves much accuracy to be desired.

#SSDs, originally developed by Google, are a balance between the two. The algorithm is more 
#straightforward (and I would argue better explained in the original seminal paper) than Faster R-CNNs.

#We can also enjoy a much faster FPS throughput than Girshick et al. at 22-46 FPS depending 
#on which variant of the network we use. SSDs also tend to be more accurate than YOLO. 
#To learn more about SSDs, please refer to Liu et al.




#When building object detection networks we normally use an existing network architecture, such 
#as VGG or ResNet, and then use it inside the object detection pipeline. The problem is that 
#these network architectures can be very large in the order of 200-500MB.

#Network architectures such as these are unsuitable for resource constrained devices due to their sheer 
#size and resulting number of computations.

#Instead, we can use MobileNets (Howard et al., 2017), another paper by Google researchers. 
#We call these networks “MobileNets” because they are designed for resource constrained devices 
#such as your smartphone. MobileNets differ from traditional CNNs through the usage of depthwise separable convolution (Figure 2 above).

#The general idea behind depthwise separable convolution is to split convolution into two stages:

#A 3×3 depthwise convolution.
#Followed by a 1×1 pointwise convolution.
#This allows us to actually reduce the number of parameters in our network.

#The problem is that we sacrifice accuracy — MobileNets are normally not 
#as accurate as their larger big brothers but they are much more resource efficient.


#HOW TO RUN
#python opencv_object_detection_more_indepth2_video.py --prototxt MobileNetSSD_deploy.prototxt.txt 
#--model MobileNetSSD_deploy.caffemodel --confidence 0.3


# import the necessary packages
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
 
# construct the argument parse and parse the arguments


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

#Lines 20-23 build a list called CLASSES  containing our labels. This is followed by a list, 
#COLORS  which contains corresponding random colors for bounding boxes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])



# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()



# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	
	
	#At this point, we have detected objects in the input frame. It is now time to look 
	#at confidence values and determine if we should draw a box + label surrounding the object
	
	
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	
	
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
	# update the FPS counter
	fps.update()
	
	
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
	
	
	
	
	
	


