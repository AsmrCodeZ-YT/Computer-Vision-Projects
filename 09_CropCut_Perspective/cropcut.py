import cv2 as cv
import numpy as np

########################################

imghight = 640
imgwight = 860
########################################
cap = cv.VideoCapture(0)
cap.set(2,imghight)
cap.set(3,imgwight)
cap.set(10, 100)


def preprocessing (img):
    
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv.dilate(imgCanny, kernel , iterations=2)
    imgThres = cv.erode(imgDial, kernel, iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxarea = 0
    
    
    contours , hierarchy = cv.findContours(img ,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 5000:
            cv.drawContours(imgContours ,cnt ,-1 ,(255,0,0) ,3 )
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            
            if area > maxarea and len(approx) == 4:
                biggest = approx
                maxarea = area
            
            
            print (len(approx))
            object = len(approx)
            x, y, w, h = cv.boundingRect(approx)
    cv.drawContours(imgContours ,biggest ,-1 ,(255,0,0) ,20)
    return biggest


def recorder (myPoint):
    myPoint = myPoint.reshape((4,2))
    myPointNew = np.zeros((4,1,2) , np.int32)
    add = myPoint.sum(1)
    # print("add" , add)
    myPointNew [0] = myPoint[np.argmin(add)]
    myPointNew [3] = myPoint[np.argmin(add)]
    diff = np.diff(myPoint,axis =1)
    myPointNew[1] = myPoint[np.argmin(diff)]
    myPointNew[2] = myPoint[np.argmin(diff)]
    # print("NewPOINT", myPointNew)
    return myPointNew

def getWarp(img,biggest):
    biggest = recorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0]]),[imgwight ,0], [0 ,imghight],[imgwight ,imghight ]
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    imgoytput = cv.warpPerspective(img, matrix, (imgwight ,imghight))  
    return imgoytput


    
while True:
    success , img = cap.read()
    img_re = cv.resize(img, (imghight,imgwight))
    
    imgContours = img.copy()
    imgThres = preprocessing(img)
    biggest = getContours(imgThres)
    print(biggest)
    imgWarped = getWarp(img, biggest)
    
    cv.imshow("videocapture", imgWarped)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break












