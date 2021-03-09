## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
##  - Tracking with hsv color bar
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import detector
import imutils
from imutils import perspective
from collections import deque

pts = deque(maxlen=64)
def nothing(x):
    pass
rx_lower = (2,35,120)
rx_upper = (85,47,130)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        bbox_obj = detector.detect(color_image,1)
        #print(bbox_obj)
        if len(bbox_obj)>0:
            x = bbox_obj[0][0]
            y = bbox_obj[0][1]
            w = bbox_obj[0][2]
            h = bbox_obj[0][3]
            
            cX = x+ w//2
            cY = y + h//2
        # Stack both images horizontally
        #images = np.hstack((color_image, depth_colormap))
            depth = depth_frame.get_distance(cX,cY)
            # dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [x,y], depth)
            # distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
            print("distance",depth*1000)

            cv2.circle(color_image, (cX,cY),5, (0,0,255),3)
        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
       

        key = cv2.waitKey(1) & 0xFF        
       
        if key==ord("q"):
            break


finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()