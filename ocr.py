from PIL import Image
import pytesseract
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
ap.add_argument("-p","--preprocess",default="thresh",help="Type of Preprocessing to be done")

args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow("original",image)
cv2.waitKey(0)
cv2.destroyWindow("original")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyWindow("gray")
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv2.imshow("thresh op",gray)
    cv2.waitKey(0)
    cv2.destroyWindow("thresh op")

elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray,3)
    cv2.imshow("blurring",gray)
    cv2.waitKey(0)
    cv2.destroyWindow("blurring")

cv2.imwrite("test.png", gray)
text = pytesseract.image_to_string(Image.open("test.png"))
os.remove("test.png")
print(text)