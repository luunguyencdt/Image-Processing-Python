# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:22:52 2020

@author: user61
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:04:21 2020

@author: user61
"""
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import glob


mtx  = np.loadtxt('cameraMatrix.txt',dtype = float)
print(mtx)
dist = np.loadtxt('dist_cofficient.txt')
print(dist)

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


#cap = cv2.VideoCapture(0)
print("[INFOR] sampling THREADED frames from webcam...")
#cap = cv2.VideoCapture(0)
cap = WebcamVideoStream(src=0).start()
fps =FPS().start()

while True:
    
    frame= cap.read()

    frame = imutils.resize(frame, width=400)    
    frame = cap.read()
    # = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    if ret ==True:
        ret,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(frame,corners,imgpts)
    cv2.imshow('img',frame)
    fps.update()    
    #print(fps.ellapsed())
    
    k = cv2.waitKey(1)  & 0xFF
    if k  == ord('q'):
            #cv2.imwrite(fname[:6]+'.jpg', img)
        break
    
fps.stop() 
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.stop()
cv2.destroyAllWindows()

 