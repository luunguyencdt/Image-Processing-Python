# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:52:56 2020

@author: Luu Nguyen
"""

import numpy as np
import cv2
# slect path to image!
img = cv2. imread('13.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 130, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
