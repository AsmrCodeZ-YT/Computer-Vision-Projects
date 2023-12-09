# owner2plusai
import cv2 as cv



# vip : CascadeClassifier so weak pretrained model for detetion 
face_detection = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture("Video.mp4")


while cap.isOpened():
    sucess, frame = cap.read()
    if not sucess:
        break

    colorRGB = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detection = face_detection.detectMultiScale(colorRGB)
    
    
    
    for (sx,sy,sw,sh) in detection:
        cv.rectangle(frame,
                     (sx,sy),
                     (sx+sw, sy+sh),
                     (255,0,0),2)
        
        cv.putText(frame,
                   "Face",
                   (sx,sy),
                   cv.FONT_HERSHEY_COMPLEX,
                   1,(0,0,255),2)
    
    cv.imshow("Result", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break 

cv.destroyAllWindows()















