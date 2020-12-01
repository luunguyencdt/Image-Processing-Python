import numpy as np
import cv2, time


cap = cv2.VideoCapture(0)
ret = cap.set(3,640)
ret = cap.set(4,480)
img_counter=200
print("reaady")
time.sleep(2)
img_counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #cv2.resize(frame, (100,100))


    # Display the resulting frame
    cv2.imshow('frame',frame)
    key =cv2.waitKey(1) & 0xFF
    if key ==ord('q'):
        break
    
    if key ==ord(' '):
        cv2.imwrite(str(img_counter)+".jpg", frame)
        print(img_counter)   
        img_counter+=1
        time.sleep(0.5)
        if img_counter ==20:
            break
    
    # cv2.imwrite('image_['+str(img_counter)+']'+".jpg", frame)
    # print("{} written".format(img_name)time.sleep(1)
    
    

   

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()