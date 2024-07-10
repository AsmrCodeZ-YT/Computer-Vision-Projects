import cv2 as cv


#rgb #bgr

img1= cv.imread("E:\Python\YoutubeFarsi\VISION\image1.jpg")
img2 = cv.imread("E:\Python\YoutubeFarsi\VISION\image2.jpg")

resize_img1 = cv.resize(img1,(1000,1000))
resize_img2 = cv.resize(img2,(1000,1000))

cv.imshow("Result img1",resize_img1)
cv.imshow("Result img2",resize_img2)
cv.waitKey(0)
# cv.destroyAllWindows()