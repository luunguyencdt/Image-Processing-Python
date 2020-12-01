# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:57:06 2020

@author: Luu Nguyen
Ver4 :
    include compare image to detected lasing
    
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:54:44 2020

@author: user61
"""

from threading import Thread
# import RPi.GPIO as GPIO
import numpy as np
import imutils
# import pigpio
import time
import math
import cv2

IMG_WIDTH=640
IMG_HEIGH=480
# Motor
stp_pin = 17 #17 #21 #Step GPIO Pin
DIR = 22 #20  #19 #Direction GPIO Pin
enbl_pin =27 #16 26

enbl_pin2 =20
DIR2 = 19
stp_pin2 = 21

# Home switch
limited = 23
limited2 = 24

CW = 1 #Clockwise Rotation
CCW =0 #Counterclockwise Rotation
SPR = 25600

home_state = 0
home_state2 = 0

global bit_home_1 
global bit_home_2
bit_home_1 = False
bit_home_2 = False

pulse_count=0
global target_pos_x_pixels
global actual_pos_x_pixels
global target_pos_y_pixels
global actual_pos_y_pixels
actual_pos_x_pixels=0
target_pos_x_pixels = 0

actual_pos_y_pixels=0
target_pos_y_pixels = 0
global target_pos_x
global actual_pos_x

global target_pos_y
global actual_pos_y
global err_pos
actual_pos_x =0
actual_pos_y =0
target_pos_x = 0
target_pos_y = 0
err_pos =0

# For galvo scaning unit is mm
d = 1150
e =28

def void_init_test():
    # Init Stepper GPIO
    #suppress warning
    GPIO.setwarnings(False)

    #use 'GPIO' pin numbering
    GPIO.setmode (GPIO.BCM)
    # Motor 1 - rectangle mirror
    GPIO.setup (enbl_pin, GPIO.OUT)
    GPIO.setup(DIR, GPIO.OUT)
    GPIO.setup(stp_pin , GPIO.OUT)
    GPIO.setup(limited,GPIO.IN)
    ### Motor 2 - circular mirror
    GPIO.setup (enbl_pin2, GPIO.OUT)
    GPIO.setup(DIR2, GPIO.OUT)
    GPIO.setup(stp_pin2 , GPIO.OUT) 
    GPIO.setup(limited2,GPIO.IN)

    GPIO.output(enbl_pin,GPIO.HIGH)
    GPIO.output(enbl_pin2,GPIO.HIGH)


def void_init():
    # Init Stepper GPIO
    #suppress warning
    GPIO.setwarnings(False)

    #use 'GPIO' pin numbering
    GPIO.setmode (GPIO.BCM)
    # Motor 1 - rectangle mirror
    GPIO.setup (enbl_pin, GPIO.OUT)
    GPIO.setup(DIR, GPIO.OUT)
    GPIO.setup(stp_pin , GPIO.OUT)
    GPIO.setup(limited,GPIO.IN)
    ### Motor 2 - circular mirror
    GPIO.setup (enbl_pin2, GPIO.OUT)
    GPIO.setup(DIR2, GPIO.OUT)
    GPIO.setup(stp_pin2 , GPIO.OUT) 
    GPIO.setup(limited2,GPIO.IN)

    GPIO.output(enbl_pin,GPIO.HIGH)
    GPIO.output(enbl_pin2,GPIO.HIGH)

    pre_run(0)
    time.sleep(1)
    pre_run(1)
    time.sleep(1)
    home_axis()
    #
    GPIO.output(enbl_pin,GPIO.HIGH)
    GPIO.output(DIR,GPIO.LOW) # dao chieu

    run_pos(0,43.7) # nho hon la positive direction
   
    GPIO.output(enbl_pin2,GPIO.HIGH)
    GPIO.output(DIR2,GPIO.LOW)
    run_pos(1,34.04)#Y


step_count = SPR


def cvt_theta2pulse(theta): #return pulse value
    return (theta*SPR)/360
def cvt_coordinate2theta(x,y):
    #print("y,d: {},{}, {}".format(y,d,y/d))
    theta_y = math.degrees(math.atan(y/d)/2)
    #print(theta_y)
    DA = math.sqrt(d**2 + y**2) + e 
    theta_x = math.degrees(math.atan(x/DA)/2)
    #print("artan : ({},{})".format(theta_x,theta_y)) # in degree
    pulse_x = cvt_theta2pulse(theta_x)
    pulse_y = cvt_theta2pulse(theta_y)
    #print("pulse: ({},{})".format(pulse_x,pulse_y))
    #run(0,pulse_x)
    #time.sleep(1)
    #run(1,pulse_y)
    return pulse_x, pulse_y
def home_axis():
    global bit_home_1 
    global bit_home_2
    # Home 
    home_state = GPIO.input(limited)
    GPIO.output(DIR,GPIO.HIGH)
    while (home_state!= 1):
        home_state = GPIO.input(limited)
        GPIO.output(stp_pin, GPIO.input(stp_pin)^1)
        time.sleep(0.0002)
        #print("homing 1 ...")
        if home_state == 1:
            GPIO.output(enbl_pin, GPIO.LOW)
            bit_home_1 = True
            print("bit_home1:",bit_home_1)
#     #Home 2
    home_state2 = GPIO.input(limited2)
    GPIO.output(DIR2,GPIO.HIGH)
    while (home_state2!= 1):
        home_state2 = GPIO.input(limited2)
        GPIO.output(stp_pin2, GPIO.input(stp_pin2)^1)
        time.sleep(0.0002)
        #print("homing y ...")
        if home_state2 == 1:
            GPIO.output(enbl_pin2, GPIO.LOW)
            bit_home_2 = True
            print("bit_home2:",bit_home_2)
            
            
            
def pre_run(x):
    for i in range(7000):
        if x ==0:
            GPIO.output(stp_pin, GPIO.input(stp_pin)^1)
            time.sleep(0.0005)
        elif x==1:
            GPIO.output(stp_pin2, GPIO.input(stp_pin2)^1)
            time.sleep(0.0005)
            
  
def run(axis,stp):
    if axis ==0:
        for pulse_count in range(stp):
            GPIO.output(stp_pin, GPIO.HIGH)
            time.sleep(0.0008)
            GPIO.output(stp_pin, GPIO.LOW)
            time.sleep(0.0008)
            pulse_count+=1
            #print("count",pulse_count)
            if pulse_count == stp:
                actual_pos_x = target_pos_x + actual_pos_x #update position
                
                pulse_count=0 #reset count
    elif axis==1:
        pulse_count=0
        if check_direction(actual_pos_y, target_pos_y)>0:
            GPIO.output(DIR2, GPIO.HIGH)
        #Keep
            if actual_pos_y!= target_pos_y and abs(stp)>3:

                for pulse_count in range(stp):
                    GPIO.output(stp_pin2, GPIO.HIGH)
                    time.sleep(0.0008)
                    GPIO.output(stp_pin2, GPIO.LOW)
                    time.sleep(0.0008)
                    pulse_count+=1
                    print("county",pulse_count)
                    if pulse_count == stp:
                        actual_pos_y = target_pos_y + actual_pos_y #update position
                        
                        pulse_count=0 #reset count
    
        
        
def check_direction(a,b):
    return (b-a)


frame = None


TrDict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'boosting': cv2.TrackerBoosting_create,
          'mil': cv2.TrackerMIL_create,
          'tld': cv2.TrackerMedianFlow_create,
          'mosse': cv2.TrackerMOSSE_create}

KNOWN_DISTANCE = 52.0 #cmq
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 2.507 #cm
#KNOWN_WIDTH = 1.863 #cm
focallength_arr=[]
def mse(imageA,imageB):
    err =np.sum(imageA.astype("float")-imageB.astype("float")**2)
    err/=float(imageA.shape[0]*imageA.shape[1])
    return err
def cross_line1(image):
    h,w,c = image.shape
    x_center = int(w/2)
    y_center = int(h/2)
    # new point rorate 4degree

    #print("(x_new,y_new)= {},{}".format(x_new,y_new))
    cv2.line(image,(x_center ,0),(x_center,h),(0,255,255))
    cv2.line(image,(0,y_center ),(w,y_center ),(0,255,255))
global line_x
global line_y
line_x = None
line_y = None

def cross_line(image):
    global line_x
    global line_y
    h,w,c = image.shape
    x_center = int(w/2)
    y_center = int(h/2)
    # new point rorate 60degree
    
    x_new = int(math.sin(math.pi/3) * y_center)
    y_new = int(math.sin(math.pi/3) * x_center)
    #print("new_x, newy: {}{}".format(x_new,y_new))
    #
    
    slope = (y_center + y_new-y_center)/(w-x_center)
    #print("slope", slope)
    # slope 0.865625

    ax = 370
    ay = slope*ax-37 # only for calibration
    ax = int(ax)
    ay = int(ay)
    line_x=((0,y_center -y_new),(w,y_center + y_new))
    line_y = (x_center + x_new,0),(x_center-x_new,h)
    # slope cuar line_y
    
    #ccc= (h)/(x_center-x_new-x_center - x_new)
    #print("line_y=",ccc)
    # ve cham tron cach 50 pixel 
    #circle(frame,(ax,ay),2, (0,255,0),2) 
    #print("(x_new,y_new)= {},{}".format(x_new,y_new))
    #cv2.line(image,(x_center + x_new,0),(x_center-x_new,h),(0,0,255))
    #cv2.line(image,(0,y_center -y_new),(w,y_center + y_new),(0,0,255))
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
    cv2.putText(frame, "TrackerCSRT: Tracking", (30,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
         
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       #raise Exception('lines do not intersect')
        x,y = (320,240) #center

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y        
def run_tracking(delta_x,delta_y):
    global actual_pos_x,actual_pos_y
    global actual_pos_x_pixels,actual_pos_y_pixels
    
    if delta_x>0:
        #print("x+")
        delta_x = delta_x*3.5
        delta_x = int(round(delta_x,5))

        GPIO.output(DIR, GPIO.HIGH)

        if actual_pos_x != int(round(target_pos_x,5)):
            for pulse_count in range(delta_x):
                GPIO.output(stp_pin, GPIO.HIGH)
                time.sleep(0.0003)
                GPIO.output(stp_pin, GPIO.LOW)
                time.sleep(0.0003)
                pulse_count+=1

                if pulse_count == delta_x:
                    actual_pos_x = round(target_pos_x,5)
                    actual_pos_x_pixels = int(round(target_pos_x_pixels,5))
                    pulse_count=0 #reset count
    elif delta_x<0:
        print("x-")
        delta_x = delta_x*3.5
        delta_x = int(round(delta_x,5))
        delta_x= abs(delta_x)
        GPIO.output(DIR, GPIO.LOW)
        if actual_pos_x != int(round(target_pos_x,5)):
            for pulse_count in range(delta_x):
                GPIO.output(stp_pin, GPIO.HIGH)
                time.sleep(0.0003)
                GPIO.output(stp_pin, GPIO.LOW)
                time.sleep(0.0003)
                pulse_count+=1

                if pulse_count == delta_x:
                    actual_pos_x = round(target_pos_x,5)
                    actual_pos_x_pixels = int(round(target_pos_x_pixels,2))
                    pulse_count=0 #reset count
        check_y = check_direction(actual_pos_y_pixels,target_pos_y_pixels) 
        print("check y",check_y)
    if delta_y >0:
    # print("y+")
        GPIO.output(DIR2, GPIO.HIGH)
        delta_y = delta_y*4.5
        delta_y = int(round(delta_y,5))
        if actual_pos_y!= int(round(target_pos_y,5)):
            for pulsey in range(delta_y):
                GPIO.output(stp_pin2, GPIO.HIGH)
                time.sleep(0.0003)
                GPIO.output(stp_pin2, GPIO.LOW)
                time.sleep(0.0003)
                pulsey+=1

                if pulsey == delta_y:
                    actual_pos_y = round(target_pos_y,5)
                    actual_pos_y_pixels = int(round(target_pos_y_pixels,5))
                    pulsey=0 #reset count
                   

    elif delta_y <0:
        print("y-")
        GPIO.output(DIR2, GPIO.LOW)
        delta_y = delta_y*4.5
        delta_y = int(round(delta_y,5))
        delta_y = abs(delta_y)
        if actual_pos_y!= int(round(target_pos_y,5)):
            for pulsey in range(delta_y):
                GPIO.output(stp_pin2, GPIO.HIGH)
                time.sleep(0.003)
                GPIO.output(stp_pin2, GPIO.LOW)
                time.sleep(0.003)
                pulsey+=1

                if pulsey == delta_y:
                    actual_pos_y = round(target_pos_y,5)
                    actual_pos_y_pixels = int(round(target_pos_y_pixels,5))
                    pulsey=0 #reset count
def run_pos(x,angle_in_degrees):
    stp= int(cvt_theta2pulse(angle_in_degrees))
    print("angle-step: {},{}".format(angle_in_degrees, stp))
    for i in range(stp*2):
        if x == 0:
            GPIO.output(stp_pin, GPIO.input(stp_pin)^1)
            time.sleep(0.0008)
        elif x == 1:
            GPIO.output(stp_pin2, GPIO.input(stp_pin2)^1)
            time.sleep(0.0008)
def run_steered(x,angle_in_degrees):
    stp= int(cvt_theta2pulse(angle_in_degrees))
    print("angle-step: {},{}".format(angle_in_degrees, stp))
    for i in range(stp*2):
        if x == 0:
            GPIO.output(stp_pin, GPIO.input(stp_pin)^1)
            time.sleep(0.0008)
        elif x == 1:
            for ix in range(stp):
                print("ix+")
                GPIO.output(DIR2, GPIO.HIGH)
                
                GPIO.output(stp_pin2, GPIO.HIGH)
                time.sleep(0.0003)
                GPIO.output(stp_pin2, GPIO.LOW)
                time.sleep(0.0003)
            for ix in range(stp):
                print("ix-")
                GPIO.output(DIR2, GPIO.LOW)
                GPIO.output(stp_pin2, GPIO.HIGH)
                time.sleep(0.0003)
                GPIO.output(stp_pin2, GPIO.LOW)
                time.sleep(0.0003)
            
x_origin_pixels = 320
y_origin_pixels = 240
actual_pos_x_pixels = x_origin_pixels
actual_pos_y_pixels = y_origin_pixels
# 
# void_init()
# if(bit_home_1==False or bit_home_2==False):
#     print("reinit")
#     void_init()  
# void_init_test()
        
camera = cv2.VideoCapture(0)
# first frame

ret = camera.set(3,IMG_WIDTH)
ret = camera.set(4,IMG_HEIGH)

#Init bbox
bbox=None

#tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerCSRT_create()

focal_len_used =  345.556
distance_arr=[]

#GPIO.output(DIR, GPIO.HIGH)
#run_pos(0,3) # nho hon la positive direction
global x_obj,y_obj
x_obj = 0
y_obj =0
global x_obj_means,y_obj_means
# x_obj_means = 0
# y_obj_means = 0
frame_count=0
nframe = 15
frames_arr_x = []
frames_arr_y = []
skip_frame =5
mtx  = np.loadtxt('cameraMatrix.txt',dtype = float)
print(mtx)
dist = np.loadtxt('dist_cofficient.txt')
print(dist)

try:
    while (camera.isOpened()) :
        #timer = cv2.getTickCount()
        grabbed, frame = camera.read()
        cross_line(frame)
        previous_frame = frame
        #frame = imutils.resize(frame, width = 600 )
        #frame = imutils.rotate(frame, 275)
        h,  w = frame.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        frame = dst
        if not grabbed:
            break
        # if the see if the roi has been computed
        if bbox is not None:    
            #bbox : (x0,y0,w,h) of an object
            success, bbox = tracker.update(frame)
            # Move to fist point
                     
            if success:
                
                drawBox(frame, bbox)
                frame_1=frame.copy()
                cut_frame =frame_1[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
                
    
                gray = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5,5), 0)
                thresh = cv2.threshold(blurred, 190 , 255, cv2.THRESH_BINARY)[1]
                cv2.imshow("thresh", thresh)
                #find the location of these white regions using contour detection
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
#                
                       #compute the center of contour
                for c in cnts:
                    if cv2.contourArea(c) > 45:
                        print("area",cv2.contourArea(c))
                        M = cv2.moments(c) 
                        #print("M00 = ", M["m00"]) 
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        x_obj = cX +int(bbox[0])
                        y_obj = cY + int(bbox[1])
                        
                        frames_arr_x.append(x_obj)
                        frames_arr_y.append(y_obj)
                        if len(frames_arr_x) == nframe:
                            x_obj_means  = sum(frames_arr_x)/ len(frames_arr_x)
                            y_obj_means  = sum(frames_arr_y)/ len(frames_arr_y)
                            x_obj= x_obj_means
                            y_obj = y_obj_means
                            print(x_obj,y_obj)
                            bias =  y_obj + x_obj*(1/0.865625) # slope of x' line rotate
                            # y = -x* (1/0.865625) +bias
                            # with y =0 ==> x
                            x_line1 = bias*0.865625                            
                            line1= ((x_obj,y_obj),(x_line1,0))
                            line2= line_x
                            ax,ay = line_intersection(line1,line2) # tim ra diem tren X' tuong ung X
                            dolon_vector_x = math.sqrt((ax-x_origin_pixels)**2+(ay-y_origin_pixels)**2) # magnatude cua X tren truc toa do motor
                            a= int(ax)
                            b = int(ay)
                            cv2.line(frame,(320,240),(a,b),(255,0,255),2)
                            #
                            bias2= y_obj - x_obj*(1/1.1594202)
                            # with y =0 ==> x
                            y_line2 = bias2
                            line1_2= ((x_obj,y_obj),(0,y_line2))
                            
                            lineY= line_y
                            ax1,ay1 = line_intersection(line1_2,lineY)
                            c = int(ax1)
                            d= int(ay1)
                            dolon_vector_y = math.sqrt((ax1-x_origin_pixels)**2+(ay1-y_origin_pixels)**2)  # toa do y tren$print("actual-taget X : {}, {}".format(actual_pos_x_pixels,x_target_pixels))
                            cv2.line(frame,(320,240),(c,d),(255,0,255),2)
                            
                                                        
                            target_pos_x_pixels = actual_pos_x_pixels + dolon_vector_x
                            target_pos_y_pixels = actual_pos_y_pixels + dolon_vector_y
                            #print("target -actual x pixels: {},{}".format(target_pos_x_pixels,actual_pos_x_pixels))
                            target_pos_x = dolon_vector_x
                            delta_x = target_pos_x -actual_pos_x
                            #print("delta x" ,delta_x)
                            
                            target_pos_y = dolon_vector_y
                            delta_y = target_pos_y-actual_pos_y
                            print("actual y-target y pixels:{},{}".format(actual_pos_y_pixels, int(round(target_pos_y_pixels,5))))
                            print("delta_y",delta_y)
                            # t = Thread (target = run_tracking, args = (delta_x,delta_y))
                            # t.start()
                            # #t.join()
                            # #t1= Thread (target = run_steered, args = (1,3))
                            # #t1.start()
                            
                            

        

                            
                            
                            
                        
                        if len(frames_arr_x) == nframe + skip_frame:
                            frames_arr_x = []
                            frames_arr_y = []
                            print("clear")
                            #t1.join()
                        
#                         cv2.drawContours(cut_frame, [c], -1, (0,255,0),2)
#                         cv2.circle(cut_frame, (cX,cY), 1, (255,255,255),-1)
#                         cv2.imshow("thresh",cut_frame)

 
            # compare frame
            
            else:
                cv2.putText(frame, "TrackerCSRT:LOST", (30,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
#                 
#             current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             current_frame_gray =cv2.GaussianBlur(current_frame_gray,(5,5),0)
#             previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
#             previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) 
#             previous_frame_gray =cv2.GaussianBlur(previous_frame_gray,(5,5),0)
#             err= mse(current_frame_gray,previous_frame_gray)
# 
#             print("MSE: {}".format(round(err,3)))

                # Calculate pixels distance and convert to real-distance
                # So xung de chay                         
                
                # Run motor here
                # conver pixels position -
                # print("cut",cut_frame)
                # if curentFrame <=200:
                #     name = path + str(curentFrame) +'.jpg'
                #     cv2.imwrite(name,cut_frame)
                #     curentFrame +=1
#                 marker = find_marker(cut_frame)
#                 
#                 # collecting 10 value of focallength
#                 # calculate avarange value
# #                 focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
# #                 if len(focallength_arr) <10 : 
# #                      focallength_arr.append(focalLength)
# #                 else:
# #                      focal_len_mean = np.sum(focallength_arr)/len(focallength_arr)
# #                      print("focal mean= " ,focal_len_mean )
# #                     
#                 distance = distance_to_camera(KNOWN_WIDTH, focal_len_used, marker[1][0])
#                 if len(distance_arr) <10:
#                     distance_arr.append(distance)
#                 else:
#                     distance_mean = np.sum(distance_arr)/len(distance_arr)
#                     print("real distance: ", distance_mean)
#                     distance_arr.clear()
#                     #cv2.putText(frame, "Distance %0.2fcm"%(distance_mean), (50,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)
#                     
#                 # ###################################################################
#         
#                           
#                 #print("distance {}".format(distance))
#             else:
#                 cv2.putText(frame, "LOST", (50,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1cross_line1(frame)
        cross_line1(frame)
        cv2.circle(frame,(320,240),2, (0,255,0),2)

        cv2.imshow("frame",frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
            # select the bouding box of the object we want to track
            # (make sure you press ENTER or SPACE after selecting ROI)
            bbox = cv2.selectROI("frame", frame, fromCenter=False, showCrosshair=True)
            
            print("Box:",bbox)
            tracker.init(frame,bbox)
            
           
        if key==ord("q"):
            # t1.join()
            break
except Exception as e:
    raise e
camera.release()
#GPIO.output(enbl_pin,GPIO.LOW)
#GPIO.output(enbl_pin2,GPIO.LOW)
# GPIO.cleanup()
cv2.destroyAllWindows()