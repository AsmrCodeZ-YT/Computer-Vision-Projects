# owner2plusai 

import cv2 as cv
from  cvzone.HandTrackingModule import HandDetector
import time



cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=2)

############################
pri_frame = 0
new_frame = 0
######################
 


while True:
    sucess, frame = cap.read()
    hand, frame = detector.findHands(frame)
    if hand:
        hand1 = hand[0]
        hmlist_r = hand1["lmList"]
        len_right ,info, frame = detector.findDistance(hmlist_r[4][:-1],hmlist_r[8][:-1],frame)
        print(len_right)
        cv.putText(frame, f"Right : {len_right:.2f}" ,(50,100),cv.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0 ),2)
    


    # show fps
    new_frame = time.time()
    fps = 1/(new_frame - pri_frame)
    pri_frame = new_frame
    cv.putText(frame, f"fps : {fps :.2f}" ,(49,50),cv.FONT_HERSHEY_SIMPLEX,1 ,(0,255,0 ),2)
    


    cv.imshow("HAND" , frame)

    if cv.waitKey(1) & 0xff == ord("q"):
        break

cv.destroyAllWindows()





