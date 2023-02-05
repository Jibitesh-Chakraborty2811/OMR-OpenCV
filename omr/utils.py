import cv2
import numpy as np

def rectContour(contours):

    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        #print(area)

        if area > 20:
            peri =cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            #print("Corner Points",len(approx))

            if len(approx) == 4:
                rectCon.append(i)

    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)

    return rectCon

def getCornerPoints(cont):
    peri =cv2.arcLength(cont,True)
    approx = cv2.approxPolyDP(cont,0.02*peri,True)
    return approx

def reorder(myPoints):
    
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    
    myPointsnew = np.zeros((4,1,2),np.int32)
    myPointsnew[0] = myPoints[np.argmin(add)]
    myPointsnew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsnew[1] = myPoints[np.argmin(diff)]
    myPointsnew[2] = myPoints[np.argmax(diff)]

    return myPointsnew

def splitboxes(img):
    cols = np.hsplit(img,12)
    cols = cols[0:4]

    return cols