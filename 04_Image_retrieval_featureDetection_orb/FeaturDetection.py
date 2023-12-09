import numpy as np
import cv2 as cv
import os


path = "./images"
orb = cv.ORB_create(nfeatures=1000)

### import images
images = []
class_names = []
my_list = os.listdir(path)
print("Total of Classes : ", len(my_list))

for cl in my_list:
    imgCur = cv.imread(f"{path}/{cl}", 0)
    images.append(imgCur)
    class_names.append(os.path.splitext(cl)[0])



def find_Des(images):
    desList = []
    for img in images:
        kp ,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def find_ID(img, desList, thres=20):
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:  
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal      
        
        
        
desList = find_Des(images)
print(len(desList))



cap = cv.VideoCapture(0)
cap.set(640,3)
cap.set(480,4)

while True:
    success ,img2 = cap.read()
    imgOriginal  = img2.copy()[80:,:]

    img2 = cv.cvtColor(img2 ,cv.COLOR_BGR2GRAY)
    
    id_ = find_ID(img2, desList)
    if id_ != -1:
        cv.putText(imgOriginal, class_names[id_] ,(50,80),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)

    cv.imshow("FeaturDetection", imgOriginal)
    if cv.waitKey(1) & 0xFF == ord("q"):break