import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX                ##Font style for writing text on video frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)        ##Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
Kernal = np.ones((3, 3), np.uint8)

mtx  = np.loadtxt('cameraMatrix.txt',dtype = float)
print(mtx)
dist = np.loadtxt('dist_cofficient.txt')
print(dist)

while(1):
    ret, frame = cap.read()         ##Read image frame
    #frame = cv2.flip(frame, +1)     ##Mirror image frame
    #frame = cv2.undistort(frame, mtx, dist, None, mtx)

    if not ret:                     ##If frame is not read then exit
        break
    if cv2.waitKey(1) == ord('s'):  ##While loop exit condition
        break
    blurred = cv2.GaussianBlur(frame,(5,5),0)
    frame2 = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)         ##BGR to HSV
    lb = np.array([18,149,0])
    ub = np.array([91,255,255])

    mask = cv2.inRange(frame2, lb, ub)                      ##Create Mask
    mask = cv2.erode(mask, None,iterations=2)
    mask = cv2.dilate(mask, None,iterations=2)
    cv2.imshow('Masked Image', mask)

    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal)        ##Morphology
    #cv2.imshow('Opening', opening)

    res = cv2.bitwise_and(frame, frame, mask= mask)             ##Apply mask on original image
    cv2.imshow('Resuting Image', res)

    (_,contours, hierarchy) = cv2.findContours(mask, cv2.RETR_TREE,      ##Find contours
                                           cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        #distance = 2*(10**(-7))* (area**2) - (0.0067 * area) + 83.487
        distance = 2*(10**(-6))* (area**2) - (0.0142 * area) + 73.758
        M = cv2.moments(cnt)
        Cx = int(M['m10']/M['m00'])
        Cy = int(M['m01'] / M['m00'])
        ##S = 'Location of object:' + '(' + str(Cx) + ',' + str(Cy) + ')'
        #cv2.putText(frame, S, (5, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        #S = 'Area of contour: ' + str(area)
        #cv2.putText(frame, S, (5, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        S = 'Distance Of Object: ' + str(distance)
        cv2.putText(frame, S, (5, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 1)
    ##Lets Detect a red ball
    cv2.imshow('Original Image', frame)
    

cap.release()                   ##Release memory
cv2.destroyAllWindows()         ##Close all the windows