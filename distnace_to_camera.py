# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 25, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 50.0 #cm
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 2.2 #cm
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
f_lens= 839.7704211148348
#f_lens= 930.6363636363635
cap = cv2.VideoCapture(0)

  
while True:
    image = cv2.imread('3.jpg')
    image = imutils.resize(image,width=600)
    marker = find_marker(image)
    #focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    # print(focalLength)
    inches = distance_to_camera(KNOWN_WIDTH, f_lens, marker[1][0])
    print("distance ",inches)
    #cv2.imshow('original',image)
    #draw a bounding box around the image and display
    box = cv2.cv.BoxPoints(marker)  if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    
    #Center point
    # start_point=(box[0][0],box[0][1])
    # end_point=(box[2][0],box[2][1])
    # w = end_point[0] - start_point[0] 
    # h = end_point[1] - start_point[1] 
    # x_center = int(start_point[0]+w/2)
    # y_center = int(start_point[1]+h/2)
    cv2.drawContours(image, [box], -1, (0,255,0))
    # #cv2.circle(image,end_point- start_point, 2, (255,0,0),4)
    # text = "("+str(x_center) +", " + str(y_center) + ") "
    # cv2.putText(image, text  ,(x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 2)
    # cv2.circle(image, (x_center, y_center), 2, (255,0,0),4)
    
    
    cv2.putText(image, "%.2fcm" % (inches),(image.shape[1] - 400, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 2)
    cv2.imshow("new", image)    

    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

# for imagePath in sorted(paths.list_images("images")):
#  	# load the image, find the marker in the image, then compute the
#  	# distance to the marker from the camera
#  	image = cv2.imread(imagePath)
#  	marker = find_marker(image)
#  	inches = distance_to_camera(KNOWN_WIDTH, f_lens, marker[1][0])
#  	# draw a bounding box around the image and display it
#  	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
#  	box = np.int0(box)
#  	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#  	cv2.putText(image, "%.2fft" % (inches / 12),
# 		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
# 		2.0, (0, 255, 0), 3)
#  	cv2.imshow("image", image)
#  	cv2.waitKey(0)
