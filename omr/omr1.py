import cv2
import numpy as np
import utils

ANSWER_KEY = [3,2]

image = cv2.imread('IMG_2010 (1).png')
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#image = cv2.resize(image,(400,400))
image = image[900:1300, 1600:3000]
imageCountours = image.copy()
window='image1'
cv2.imshow(window,image)
cv2.waitKey(0)

imgGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

contours, heirarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imageCountours,contours,-1,(0,255,0),5)
cv2.imshow(window,imageCountours)
cv2.waitKey(0)

rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
print(len(rectCon))
print(biggestContour)
secondbiggest = utils.getCornerPoints(rectCon[1])
cv2.drawContours(imageCountours,biggestContour,-1,(255,0,0),10)
cv2.drawContours(imageCountours,secondbiggest,-1,(100,0,0),10)
cv2.imshow(window,imageCountours)
cv2.waitKey(0)

biggestContour = utils.reorder(biggestContour)
pt1 = np.float32(biggestContour)
pt2 = np.float32([[0,0],[400,0],[0,1400],[400,1400]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored1 = cv2.warpPerspective(image,matrix,(400,1400))
imgWarpColored1 = cv2.resize(imgWarpColored1,(600,200))
cv2.imshow(window,imgWarpColored1)
cv2.waitKey(0)

secondbiggest = utils.reorder(secondbiggest)
pt1 = np.float32(secondbiggest)
pt2 = np.float32([[0,0],[400,0],[0,1400],[400,1400]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored2 = cv2.warpPerspective(image,matrix,(400,1400))
imgWarpColored2 = cv2.resize(imgWarpColored2,(600,200))
cv2.imshow(window,imgWarpColored2)
cv2.waitKey(0)

imgWarpGray1 = cv2.cvtColor(imgWarpColored1,cv2.COLOR_BGR2GRAY)
imgThresh1 = cv2.threshold(imgWarpGray1,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh1)
cv2.waitKey(0)

imgWarpGray2 = cv2.cvtColor(imgWarpColored2,cv2.COLOR_BGR2GRAY)
imgThresh2 = cv2.threshold(imgWarpGray2,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh2)
cv2.waitKey(0)

set1 = utils.splitboxes(imgThresh1)
set2 = utils.splitboxes(imgThresh2)

detect = []

max1 = 0
for i in range(0,4):
    if cv2.countNonZero(set1[i]) >= cv2.countNonZero(set1[max1]):
        max1 = i
detect.append(max1)

max2 = 0
for i in range(0,4):
    if cv2.countNonZero(set2[i]) >= cv2.countNonZero(set2[max2]):
        max2 = i
detect.append(max2)

count = 0
for i in range(0,2):
    if ANSWER_KEY[i] == detect[i]:
        count = count + 1

print(count)
print(detect)