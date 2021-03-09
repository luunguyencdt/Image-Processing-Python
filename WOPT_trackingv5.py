# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:57:06 2020

@author: Luu Nguyen

Version 5
update rotate axes function

"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:54:44 2020

@author: user61
"""

import cv2
import imutils
import numpy as np
import math
import detector

frame = None
roiPts = []
inputMode = False
bbox=[]



def rotate_function_point(x,y,theta):
    
    radian = theta*math.pi/180.0 # degree to radian

    x_new = (x-320)*math.cos(radian) + (y-240)*math.sin(radian)
    y_new = (y-240)*math.cos(radian) - (x-320)*math.sin(radian)

    x_new = x_new +320
    y_new = y_new + 240
    # trong truc toa do uv
    return int(round(x_new)),int((round(y_new)))
def inverse_rotate(x,y,theta): #real data
    
    radian = theta*math.pi/180.0 # degree to radian

    # x_new =(x-320)*math.cos(radian) - (y-240)*math.sin(radian)
    # y_new = (x-320)*math.sin(radian) +(y-240)*math.cos(radian)
    x_new = (x-320)*math.cos(radian)-(y-240)*math.sin(radian)
    y_new = (x-320)*math.sin(radian) + (y-240)*math.cos(radian)

    # trong truc toa do uv
    x_new = x_new +320
    
    y_new = y_new +240

    return (int(round(x_new)),int(round(y_new)))
    #return int(x_new+320),int(y_new+240)
def rotate_function_axis(x,y,theta):
    
    radian = theta*math.pi/180.0 # degree to radian

    x_new =(x)*math.cos(radian) - (y)*math.sin(radian)
    y_new = (x)*math.sin(radian) +(y)*math.cos(radian)
    #print("x new , y new: {},.{}".format(x_new, y_new))
    # trong truc toa do uv
    x_new = x_new
    y_new = y_new 

    return (int(round(x_new)),int(round(y_new)))
    #return int(x_new+320),int(y_new+240)

def cross_line(image):
    h,w,c = image.shape
    x_center = int(w/2)
    y_center = int(h/2)
    # new point rorate 4degree

    #print("(x_new,y_new)= {},{}".format(x_new,y_new))
    cv2.line(image,(x_center ,0),(x_center,h),(0,0,255))
    cv2.line(image,(0,y_center ),(w,y_center ),(0,0,255))
    
def cross_line_rotate(image):
    h,w,c = image.shape
    # x_center = int(w/2)
    # y_center = int(h/2)
    x_center = 320
    y_center = 240
    # new point rorate 4degree
    x_new,y_new = rotate_function_point(600,240,60) # x axis to u axis counter clockwise
    x_new1,y_new1 = rotate_function_point(320,480,60) # y axis to v axis
    cv2.line(image,(320,240),(x_new,y_new),(0,100,255))
    cv2.line(image,(320,240 ),(x_new1,y_new1 ),(0,100,255))



def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)), (0,0,255),1,1)
    #cv2.polylines(img,[bbox], (0,255,0),2)
    cv2.putText(frame, "Tracking", (50,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode
    
    # if we are in ROi selection, the mouse was cliceds
    # and we do not already have four pouints, then update the
    # list of ROi points with the (x,y) location of the clicj
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) <4:
        roiPts.append((x,y))
        cv2.circle(frame,(x,y),2, (0,255,0),2)
        print("selected points: " , roiPts)
        cv2.imshow("frame",frame)
def nothing():
    pass        
camera = cv2.VideoCapture(0)    

cv2.namedWindow("Tracking")    
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)   
cv2.createTrackbar("Area", "Tracking", 0, 1000, nothing)   
roiBox = None
previousTrackbarValue = -1  # S
mtx  = np.loadtxt('cameraMatrix.txt',dtype = float)
print(mtx)
dist = np.loadtxt('dist_cofficient.txt')
print(dist)
#tracker = cv2.TrackerCSRT_create()
(cX,cY) =(466,284)
try:
    while True:
        #timer = cv2.getTickCount()
        grabbed, frame = camera.read()
        #frame = cv2.undistort(frame, mtx, dist, None, mtx)
        
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        
        l_v = cv2.getTrackbarPos("LV", "Tracking")
         
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        area = cv2.getTrackbarPos("Area","Tracking")
         

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])
        

        # cv2.circle(frame, (cX,cY), 1, (0,255,255),4)
        # cX1,cY1 = rotate_function_point(cX,cY,60) # 
        # cv2.circle(frame, (cX1,cY1), 5, (0,0,255),4)
        
        
        # cX2,cY2 = inverse_rotate(cX1,cY1,60) # 
        # cv2.circle(frame, (cX2,cY2), 1, (255,0,255),4)
        #print("shape",frame.shape)
        #frame = imutils.resize(frame, width = 600 )
        bbox_obj = detector.detect(frame,1,l_b,u_b,area)

        #print(bbox_obj)
        if len(bbox_obj)>0:
            x = bbox_obj[0][0]
            y = bbox_obj[0][1]
            w = bbox_obj[0][2]
            h = bbox_obj[0][3]
            
            cX = x+ w//2
            cY = y + h//2
           
            # simulaion part 
            # project an object onto new coordinate 
            # by vector method
            u1,u2 = rotate_function_point(640,240,60) # x axis to u axis counter clockwise
            cv2.line(frame,(320,240),(u1,u2),(255,255,0),1)
            
            v1,v2 = rotate_function_point(320,480,60)    
            cv2.line(frame,(320,240),(v1,v2),(255,255,0),1)
            
            u1 = u1-320
            u2 = u2 -240
            
            # project Cx1,cy1 to du,dv
            a = (cX-320)*(u1) + (cY-240)*(u2)
           
            under = math.sqrt((u1**2+u2**2))**2
            unit_a = a/under # unit vector

            magnetude_a =a// math.sqrt((u1**2+u2**2))
            #print ("unit a %s" % magnetude_a)

            cx1_u = int(round(unit_a *u1)) + 320
            cy1_u = int(round(unit_a *u2)) + 240
            cv2.circle(frame, (cx1_u,cy1_u), 2, (100,0,255),6)
            
            v1 = v1 -320
            v2 = v2 - 240
            
            b = (cX-320)*(v1) + (cY-240)*(v2)
            under1 = math.sqrt((v1**2+v2**2))**2
            unit_b = b/under1
            
            cx1_v = int(round(unit_b *v1)) + 320
            cy1_v = int(round(unit_b *v2)) + 240


            
            cv2.circle(frame, (cx1_v,cy1_v), 2, (100,0,255),6)
            
            cv2.circle(frame, (cX,cY), 5, (100,255,255),2)
            
            cv2.line(frame, (cX,cY), (cx1_u,cy1_u), (200,255,0),1)
            cv2.line(frame, (cX,cY), (cx1_v,cy1_v), (200,255,0),1)

        cross_line(frame)
        #cross_line_rotate(frame)
    
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF        
       
        if key==ord("q"):
            break
except Exception as e:
    raise e
camera.release()
cv2.destroyAllWindows()