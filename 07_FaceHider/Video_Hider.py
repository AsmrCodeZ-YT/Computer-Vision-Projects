from cvzone.FaceDetectionModule import FaceDetector 
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(3,480)
detector = FaceDetector(minDetectionCon=0.75)

while True:
    success1 , img = cap.read()
    success2 , realImg = cap.read()
    
    if not success1:
        break
    if not success2:
        break 
    
    img = cv.resize(img,(640,480))
    realImg = cv.resize(realImg,(640,480))

    # ai model to detect face 
    img , bboxs = detector.findFaces(img,draw=True)
 
    if bboxs:
        for i, bbox in enumerate(bboxs):
            x,y,w,h = bbox["bbox"]
            if x < 0: x = 0
            if y < 0: y = 0
            #blur image and overwrite on real image
            imgCrop = img[y:y+h , x:x+w]
            imgBlur = cv.blur(imgCrop ,(35,35))
            img[y:y+h , x:x+w] = imgBlur    
    
    # stack image
    im_h = cv.hconcat([realImg, img])
    # remove watermark 
    best = im_h[40: , :] # not important
    cv.imshow("IMAGE", im_h)
    if cv.waitKey(1) & 0xff == ord("q"):
        break