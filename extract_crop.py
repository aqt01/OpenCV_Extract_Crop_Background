import numpy as np
import cv2 
from PIL import Image
import os

# Iterating in folder to extract and crop from all files

dirr = './Prueba/'

name=[]

for subdir, dirs, files in os.walk(dirr):
    for file in files:
        name.append( (os.path.join(subdir,file)) )

# Shows and crops the given imagen 
def crop_show(name):
    print name
    im = cv2.imread(name)

    # Median Blur, used in case of noisy images
    #im = cv2.medianBlur(im,5)

    hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    COLOR_MIN = np.array([10, 10, 10],np.uint8)
    COLOR_MAX = np.array([255, 255, 255],np.uint8)
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    imgray = frame_threshed
    ret,thresh = cv2.threshold(frame_threshed,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    x,y,w,h = cv2.boundingRect(cnt)
    
    # Draws a green rectangle on object to crop
    #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    print x
    print y
    print h
    print w

    crop_img = im[y:y+h+5,x:x+w+5]
    # Shows Raw and Cropped image
    """
    cv2.imshow("RawImage",im)
    cv2.imshow("CroppedImage",crop_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    return crop_img
    
for i in name:
    if (i[-3:]=="jpg" or i[-3:]=="JPG"):
        crop_show(i)
