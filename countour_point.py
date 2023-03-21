# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:52:02 2020
OPENCV Center of contour
https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/

https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
@author: Luu

https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
Arclen - Number of point on the contours
"""

#import the necessary packages
import argparse
import imutils
import cv2
import time
import numpy as np
# path = "D:/Luu/Image Processing Python/5. center of contour/shapes_and_colors.jpg"
# #construct the argument parse and parse the argument
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--D:/Luu/Image Processing Python/5. center of contour/" , required = True , help = path)
# args = vars(ap.parse_args())
MIN_THRESH  = 50 
font = cv2.FONT_HERSHEY_COMPLEX
#load the image 
# image = cv2.imread(args["image"])
image = cv2.imread("shapes_and_colors.jpeg")

#convert to grayscale, blur it slightly
#and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60 , 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
#find the location of these white regions using contour detection
_, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#contours = contours[0] if len(contours) == 2 else contours[3]

#loop over the contours 
for cnt in contours: 
    # Loop over epsilon size
    #for esp in np.linspace(0.001, 0.05,100):    
    approx = cv2.approxPolyDP(cnt, 0.0001  * cv2.arcLength(cnt, True), True)
# draws boundary of contours.
    cv2.drawContours(image, [approx], 0, (0, 0, 255),1)
    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    n = approx.ravel()  
    print(n)
    print(len(n))
    i = 0        
    
    if(len(n)>2):
        for j in n :        
            if (i%2 == 0):
                cv2.circle(image, (n[i],n[i + 1]), 2, (255,255,255),-1)
            #cv2.circle(image, (n[2],n[3]), 2, (255,0,255),-1)
            #cv2.circle(image, (n[4],n[5]), 2, (0,255,255),-1)
            i = i + 1
            
    cv2.imshow("Image",image)  
    cv2.waitKey(0)
cv2.destroyAllWindows() 
