import numpy as np
import cv2 as cv


img1 = cv.imread("./images/eskelet.jpg",0)
img2 = cv.imread("./images/1.jpg",0)
# img3 = cv.imread("./3.jpg")


orb = cv.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# imgkp1 = cv.drawKeypoints(img1 , kp1, None) 
# imgkp2 = cv.drawKeypoints(img2 , kp2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2 ,k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2,good ,None ,flags=2)
# cv.imshow("imgk1",imgkp1)
# cv.imshow("imgk2",imgkp2)
print(len(good))
cv.imshow("img1",img1)
cv.imshow("img2",img2)
cv.imshow("img3",img3)

cv.waitKey(0)














