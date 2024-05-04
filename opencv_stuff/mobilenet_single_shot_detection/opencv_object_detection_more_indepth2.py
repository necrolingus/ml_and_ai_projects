#https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/
#https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
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
#python opencv_object_detection_more_indepth2.py --prototxt MobileNetSSD_deploy.prototxt.txt 
#--model MobileNetSSD_deploy.caffemodel --image images/example_01.jpg 


# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
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



# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()


#We start by looping over our detections, keeping in mind that multiple 
#objects can be detected in a single image. We also apply a check to the 
#confidence (i.e., probability) associated with each detection. If the 
#confidence is high enough (i.e. above the threshold), then we’ll display 
#the prediction in the terminal as well as draw the prediction on the 
#image with text and a colored bounding box


# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]
 
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
