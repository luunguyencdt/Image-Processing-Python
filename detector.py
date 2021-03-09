# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:53:53 2021

@author: Detection methods for OWPT
"""

import numpy as np
import imutils
import cv2

# for depth camera


def detect(frame, debugMode,lower_val,upper_val,Area):
    #(x,y,w,h) =(0,0,0,0)
    blurred = cv2.GaussianBlur(frame,(5,5),0)
    
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask =  cv2.inRange(hsv,lower_val,upper_val )
    
    mask = cv2.erode(mask, None,iterations=2)
    mask = cv2.dilate(mask, None,iterations=2)
        
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    
    if (debugMode):
        cv2.imshow("mask",mask)
        cv2.imshow("res",res)

    bbox =[]

    cnts,hierarchy=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
       
        if cv2.contourArea(c)<Area:
            continue
        
        #bounding box for contour
       
       #y = 3E-05x2 - 0.2355x + 650.89
        #distance_mm = 3*(10**-5)* (area**2) - (0.2355*area) + 650.89
        # print("distance : ",distance_mm)

        (x,y,w,h) = cv2.boundingRect(c)

        bbox.append(([x,y,w,h]))

    return bbox
        
lower_yellow = (25,64,146)
upper_yellow = (45,179,255)
minArea =1500             
def detect_one(frame, debugMode):
    #(x,y,w,h) =(0,0,0,0)
    blurred = cv2.GaussianBlur(frame,(5,5),0)
    
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask =  cv2.inRange(hsv,lower_yellow,upper_yellow )
    
    mask = cv2.erode(mask, None,iterations=2)
    mask = cv2.dilate(mask, None,iterations=2)
        
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    
    if (debugMode):
        cv2.imshow("mask",mask)
        cv2.imshow("res",res)

    bbox =[]
    (_,cnts,_)=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c)<minArea:
            continue
        
        # bounding box for contour
        (x,y,w,h) = cv2.boundingRect(c)

        bbox.append(([x,y,w,h]))

    return bbox
        


# cap = cv2.VideoCapture(0)


# while True:
#     _,frame = cap.read()

#     centers = detect(frame,1)

#     if len(centers)>0:
#        #print("data",centers[0][0])
#        cv2.rectangle(frame, (centers[0][0],centers[0][1]), 
#                  (centers[0][0]+centers[0][2],centers[0][1]+centers[0][3]),
#                  (0,255,255),2)
       

       

    
#     # blurred = cv2.GaussianBlur(frame,(5,5),0)
    
#     # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
#     # mask = cv2.inRange(hsv,lower_yellow,upper_yellow )
    
#     # mask = cv2.dilate(mask, None, iterations=2)
    
#     # res = cv2.bitwise_and(frame,frame, mask= mask)
#     # (_,cnts,_)=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     # for c in cnts:
#     #     if cv2.contourArea(c)<minArea:
#     #         continue
        
#     #     # bounding box for contour
#     #     (x,y,w,h) = cv2.boundingRect(c)
#     #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
#     #     # M = cv2.moments(c)
#     #     # cX = int(M["m10"]/M["m00"])
#     #     # cY = int(M["m01"]/M["m00"])
        
#     #     # centers.append(np.array([[cX],[cY]]))
        
        
 
        

#     cv2.imshow("frame",frame)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break
    
    
# cap.release()
# cv2.destroyAllWindows()



    
