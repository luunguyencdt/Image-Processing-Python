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

frame = None
roiPts = []
inputMode = False
bbox=[]

TrDict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'boosting': cv2.TrackerBoosting_create,
          'mil': cv2.TrackerMIL_create,
          'tld': cv2.TrackerMedianFlow_create,
          'mosse': cv2.TrackerMOSSE_create}

KNOWN_DISTANCE = 50.0 #cm
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 2.507 #cm
focallength_arr=[]

def cross_line(image):
    h,w,c = image.shape
    x_center = int(w/2)
    y_center = int(h/2)
    # new point rorate 4degree

    #print("(x_new,y_new)= {},{}".format(x_new,y_new))
    cv2.line(image,(x_center ,0),(x_center,h),(0,255,255))
    cv2.line(image,(0,y_center ),(w,y_center ),(0,255,255))
    
def cross_line_rotate(image):
    h,w,c = image.shape
    x_center = int(w/2)
    y_center = int(h/2)
    # new point rorate 4degree

    #print("(x_new,y_new)= {},{}".format(x_new,y_new))
    cv2.line(image,(x_center ,0),(x_center,h),(0,255,255))
    cv2.line(image,(0,y_center ),(w,y_center ),(0,255,255))
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 25, 125)
    cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
    return cv2.minAreaRect(c)
    

    
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

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
        
camera = cv2.VideoCapture(0)    
cv2.namedWindow("frame")    
cv2.setMouseCallback("frame",selectROI)    
roiBox = None
tracker = cv2.TrackerCSRT_create()
curentFrame=0
cut_frame =None
path='D:/0.HaLab_Project/WOPT Project/10.ML_objectdetection/Obj_Tracking/trainingdata/frame'
focal_len_used = 454.726765057838
distance_arr=[]
try:
    while True:
        #timer = cv2.getTickCount()
        grabbed, frame = camera.read()
        #frame = imutils.resize(frame, width = 600 )
        cross_line(frame)
    
    
        if not grabbed:
            break
        # if the see if the roi has been computed
        if roiBox is not None:    
            tracker.init(frame,roiBox)
            success, bbox = tracker.update(frame)
            if success:
                drawBox(frame, bbox)
                frame_1=frame.copy()
                cut_frame =frame_1[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
                # print("cut",cut_frame)
                # if curentFrame <=200:
                #     name = path + str(curentFrame) +'.jpg'
                #     cv2.imwrite(name,cut_frame)
                #     curentFrame +=1
                marker = find_marker(cut_frame)
                
                # collecting 10 value of focallength
                # calculate avarange value
                # focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
                # if len(focallength_arr) <10 : 
                #     focallength_arr.append(focalLength)
                # else:
                #     focal_len_mean = np.sum(focallength_arr)/len(focallength_arr)
                #     print("focal mean= " ,focal_len_mean )
                    
                distance = distance_to_camera(KNOWN_WIDTH, focal_len_used, marker[1][0])
                if len(distance_arr) <10:
                    distance_arr.append(distance)
                else:
                    distance_mean = np.sum(distance_arr)/len(distance_arr)
                    print("real distance: ", distance_mean)
                    distance_arr.clear()
                    #cv2.putText(frame, "Distance %0.2fcm"%(distance_mean), (50,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)
                    
                # ###################################################################
        
                          
                #print("distance {}".format(distance))
            else:
                cv2.putText(frame, "LOST", (50,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                
    
    
    
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("i") and len(roiPts)<4:
            inputMode = True
            orig = frame.copy()
            while len(roiPts)<4:
                cv2.imshow("frame",frame)
                cv2.waitKey(0)
                
            pre_roiBox = np.array(roiPts)
            s = pre_roiBox.sum(axis = 1)
            print("sum,", s)
            tl = pre_roiBox[np.argmin(s)]
            print("tl: {}".format(tl))
            br = pre_roiBox[np.argmax(s)]
            print("br: {}".format(br))
            # grab roi for the bounding box
            roi = orig[tl[1]:br[1], tl[0]: br[0]]
            name = path + "aaaaaa" +'.jpg'
            cv2.imwrite(name,roi)
            roiBox = (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])
            
            print("roiBox:",roiBox)
        if key==ord("q"):
            break
except Exception as e:
    pass
camera.release()
cv2.destroyAllWindows()