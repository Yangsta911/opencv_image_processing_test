import cv2 as cv
import numpy as np
import pytesseract
from imutils.perspective import four_point_transform
from imutils import contours

img = cv.imread("Test_Image_1.jpg")
cv.imshow("Display window", img)
k = cv.waitKey(0)

cropped_image = img[165:195, 134:175]
cv.imshow("cropped", cropped_image)
k = cv.waitKey(0)
hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)

# Get binary-mask
msk = cv.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
krn = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
dlt = cv.dilate(msk, krn, iterations=1)
thr = 255 - cv.bitwise_and(dlt, msk)


cv.imshow("final", thr)
k = cv.waitKey(0)

result = pytesseract.image_to_string(thr, config="--psm 10")
print(result)
