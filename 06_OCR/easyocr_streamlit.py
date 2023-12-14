import easyocr 
import cv2 as cv

img = cv.imread("./1.png")
reader = easyocr.Reader(["en"],
                        gpu=True)
result = reader.readtext(img,
                        detail=0,
                    paragraph=True)

print(result)