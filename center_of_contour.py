# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:52:02 2020
OPENCV Center of contour
https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/

https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
@author: user61
"""

#import the necessary packages
import argparse
import imutils
import cv2
import time
# path = "D:/Luu/Image Processing Python/5. center of contour/shapes_and_colors.jpg"
# #construct the argument parse and parse the argument
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--D:/Luu/Image Processing Python/5. center of contour/" , required = True , help = path)
# args = vars(ap.parse_args())
MIN_THRESH  = 50 
#load the image 
# image = cv2.imread(args["image"])
image = cv2.imread("shapes_and_colors.jpg")

#convert to grayscale, blur it slightly
#and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

#find the location of these white regions using contour detection
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


#loop over the contours
for c in cnts:
    if cv2.contourArea(c) > MIN_THRESH:
        print(cv2.contourArea(c))
        #compute the center of contour
        M = cv2.moments(c) 
        #print("M00 = ", M["m00"]) 
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    #draw the contour and center of the shape on image
        cv2.drawContours(image, [c], -1, (0,255,0),2)
        cv2.circle(image, (cX,cY), 7, (255,255,255),-1)
        cv2.putText(image, "center", (cX-20,cY-20), cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255,255,255),2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)
cv2.destroyAllWindows()