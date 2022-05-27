import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture('video.mp4')

subtractor = cv2.bgsegm.createBackgroundSubtractorMOG() #creating the backgroung subtractor object
kernalOp = np.ones((3,3),np.uint8)

while (1):
    if (cap.isOpened()== False): 
      print("Error opening video file")
      break
    ret , frame1 = cap.read()
    if ret!= True:
      break
  
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #creating a grayscaled image from each frame
    blur = cv2.GaussianBlur(grey,(3,3),5) # blurring the grey image
    
    img_sub = subtractor.apply(blur) # background foreground separation aplying bgsegm
  
    #Closing i.e First Dilate then Erode
    #It is useful in closing small holes inside the foreground objects, or small black points on the object
    mask=cv2.morphologyEx(img_sub,cv2.MORPH_CLOSE,kernalOp)

    mask2=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalOp)

    #Finding Contours
    _, contours, _=cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for(i,cnt) in enumerate(contours): #proccesing every found contour 
        area=cv2.contourArea(cnt)
        x,y,width,height=cv2.boundingRect(cnt)

        if (area > 128*128) and height <= 1.3*width and height >=0.7*width: #area thresholding to get rid of small sized and non (almost) rectangle shaped contours
       
          img = cv2.rectangle(frame1,(x,y),(x+width,y+height),(0,255,0),2) # drawing a rectangle around the selected moving object  
        
    cv2.imshow('car',frame1) #displaying each frame  

    if cv2.waitKey(1) == 27: #waiting 1ms for user pressed ESC button
              break

cv2.destroyAllWindows()
cap.release()          