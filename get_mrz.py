from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import pytesseract as tess

# get the image address
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())

# initializing a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# image = cv2.imread(args["images"])
image = cv2.imread("chq_demo.jpg")
image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blurring for smoothing image
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# morphological operation for finding dark region in light background
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
cv2.imshow("sobel", gradX)
cv2.waitKey(0)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

# applying a closing operation using rectangular kernel to close gaps in between letters
# then apply otsu threshold method

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2.imshow("close rect", gradX)
cv2.waitKey(0)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("OTSU threshold", thresh)
cv2.waitKey(0)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow("Vertical Grouping", thresh)
cv2.waitKey(0)
thresh = cv2.erode(thresh, None, iterations=5)
cv2.imshow("Eroding", thresh)
cv2.waitKey(0)

cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
boundries = cv2.drawContours(image.copy(), cnts, -1, (0, 255, 0), 3)
cv2.imshow("contours", boundries)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#
for c in cnts:
    padding = 10
    (x, y, w, h) = cv2.boundingRect(c)
    print("{}---{}".format(w, h))
    if w > 100 and h > 10:
        # img = cv2.rectangle(image, (x - 0.3, y - 0.3), (x + w + 0.3, y + h + 0.3), (124, 201, 98), 2)
        x1 = int(x-padding)
        y1 = int(y-padding)
        x2 = int(x+w+padding)
        y2 = int(y+h+padding)

        roi = image[y1:y2,x1:x2]
        cv2.imshow('selected area',roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = cv2.rectangle(image, (x1 , y1), (x2 , y2), (124, 201, 98), 2)

        text = tess.image_to_string(roi)
        print(text)


cv2.imshow("final",image)
cv2.waitKey(0)
