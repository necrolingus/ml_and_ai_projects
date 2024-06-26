import numpy as np
import cv2

#building your own cascades https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('cascade_face.xml')
eye_cascade = cv2.CascadeClassifier('cascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascade_smile.xml')


cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	#cv2.putText(faces, str('koos'), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
	
	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		cv2.putText(roi_color, str('face'), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv2.putText(roi_color, str('eye'), (ex, ey - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


		#smile is way too sensitive, but its OK for now
		# smiles = smile_cascade.detectMultiScale(roi_gray)
		# for (sx,sy,sw,sh) in smiles:
			# cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
			# cv2.putText(roi_color, str('smile'), (sx, sy - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
	
	
	
	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()