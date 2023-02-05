
import cv2
import numpy as np
import utils


ANSWER_KEY = [1,3,0,0,2]

image = cv2.imread('IMG_2010 (1).png')
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
print(image.shape)
image = image[600:2500, 300:1500] 
#image = cv2.resize(image,(800,300))
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
thirdbiggest = utils.getCornerPoints(rectCon[2])
fourthbiggest = utils.getCornerPoints(rectCon[3])
fifthbiggest = utils.getCornerPoints(rectCon[4])
cv2.drawContours(imageCountours,biggestContour,-1,(255,0,0),10)
cv2.drawContours(imageCountours,secondbiggest,-1,(100,0,0),10)
cv2.drawContours(imageCountours,thirdbiggest,-1,(200,0,0),10)
cv2.drawContours(imageCountours,fourthbiggest,-1,(0,100,0),10)
cv2.drawContours(imageCountours,fifthbiggest,-1,(0,200,0),10)
cv2.imshow(window,imageCountours)
cv2.waitKey(0)

biggestContour = utils.reorder(biggestContour)
pt1 = np.float32(biggestContour)
pt2 = np.float32([[0,0],[1900,0],[0,1200],[1900,1200]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored1 = cv2.warpPerspective(image,matrix,(1900,1200))
imgWarpColored1 = cv2.resize(imgWarpColored1,(600,200))
cv2.imshow(window,imgWarpColored1)
cv2.waitKey(0)

secondbiggest = utils.reorder(secondbiggest)
pt1 = np.float32(secondbiggest)
pt2 = np.float32([[0,0],[1900,0],[0,1200],[1900,1200]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored2 = cv2.warpPerspective(image,matrix,(1900,1200))
imgWarpColored2 = cv2.resize(imgWarpColored2,(600,200))
cv2.imshow(window,imgWarpColored2)
cv2.waitKey(0)

thirdbiggest = utils.reorder(thirdbiggest)
pt1 = np.float32(thirdbiggest)
pt2 = np.float32([[0,0],[1900,0],[0,1200],[1900,1200]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored3 = cv2.warpPerspective(image,matrix,(1900,1200))
imgWarpColored3 = cv2.resize(imgWarpColored3,(600,200))
cv2.imshow(window,imgWarpColored3)
cv2.waitKey(0)

fourthbiggest = utils.reorder(fourthbiggest)
pt1 = np.float32(fourthbiggest)
pt2 = np.float32([[0,0],[1900,0],[0,1200],[1900,1200]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored4 = cv2.warpPerspective(image,matrix,(1900,1200))
imgWarpColored4 = cv2.resize(imgWarpColored4,(600,200))
cv2.imshow(window,imgWarpColored4)
cv2.waitKey(0)

fifthbiggest = utils.reorder(fifthbiggest)
pt1 = np.float32(fifthbiggest)
pt2 = np.float32([[0,0],[1900,0],[0,1200],[1900,1200]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored5 = cv2.warpPerspective(image,matrix,(1900,1200))
imgWarpColored5 = cv2.resize(imgWarpColored5,(600,200))
cv2.imshow(window,imgWarpColored5)
cv2.waitKey(0)

imgWarpGray1 = cv2.cvtColor(imgWarpColored1,cv2.COLOR_BGR2GRAY)
imgThresh1 = cv2.threshold(imgWarpGray1,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh1)
cv2.waitKey(0)

imgWarpGray2 = cv2.cvtColor(imgWarpColored2,cv2.COLOR_BGR2GRAY)
imgThresh2 = cv2.threshold(imgWarpGray2,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh2)
cv2.waitKey(0)

imgWarpGray3 = cv2.cvtColor(imgWarpColored3,cv2.COLOR_BGR2GRAY)
imgThresh3 = cv2.threshold(imgWarpGray3,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh3)
cv2.waitKey(0)

imgWarpGray4 = cv2.cvtColor(imgWarpColored4,cv2.COLOR_BGR2GRAY)
imgThresh4 = cv2.threshold(imgWarpGray4,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh4)
cv2.waitKey(0)

imgWarpGray5 = cv2.cvtColor(imgWarpColored5,cv2.COLOR_BGR2GRAY)
imgThresh5 = cv2.threshold(imgWarpGray5,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow(window,imgThresh5)
cv2.waitKey(0)

set1 = utils.splitboxes(imgThresh1)
set2 = utils.splitboxes(imgThresh2)
set3 = utils.splitboxes(imgThresh3)
set4 = utils.splitboxes(imgThresh4)
set5 = utils.splitboxes(imgThresh5)

print(cv2.countNonZero(set1[0]))
print(cv2.countNonZero(set1[1]))

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

max3 = 0
for i in range(0,4):
    if cv2.countNonZero(set3[i]) >= cv2.countNonZero(set3[max3]):
        max3 = i
detect.append(max3)

max4 = 0
for i in range(0,4):
    if cv2.countNonZero(set4[i]) >= cv2.countNonZero(set4[max4]):
        max4 = i
detect.append(max4)

max5 = 0
for i in range(0,4):
    if cv2.countNonZero(set5[i]) >= cv2.countNonZero(set5[max5]):
        max5 = i
detect.append(max5)

count = 0
for i in range(0,5):
    if ANSWER_KEY[i] == detect[i]:
        count = count + 1

print(count)
print(detect)