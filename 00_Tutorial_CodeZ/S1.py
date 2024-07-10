import cv2 as cv


cap = cv.VideoCapture("./1.mp4")


while True:
    success ,frame = cap.read()
    
    if not success:
        break
    
    
    
    
    
    
    cv.imshow("movie",frame)
    if cv.waitKey(25) & 0XFF == ord("q"):
        break
    
    
cv.destroyAllWindows()