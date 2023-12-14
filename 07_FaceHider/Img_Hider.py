from cvzone.FaceDetectionModule import FaceDetector 
import cv2 as cv

detector = FaceDetector(minDetectionCon=0.7)

img = cv.imread("./img3.jpg")
realImg = img.copy()

img = cv.resize(img ,(500,700))
realImg = cv.resize(realImg ,(500,700))
print(img.shape)
print(realImg.shape)

img , bboxs = detector.findFaces(img,draw=True)

print(bboxs)
if bboxs:
        for i, bbox in enumerate(bboxs):
            x,y,w,h = bbox["bbox"]
            if x < 0: x = 0
            if y < 0: y = 0
            #blur image and overwrite on real image
            imgCrop = img[y:y+h , x:x+w]
            imgBlur = cv.blur(imgCrop ,(35,35))
            img[y:y+h , x:x+w] = imgBlur    

im_h = cv.hconcat([realImg, img])
cv.imshow("Result",im_h)
cv.waitKey(0)