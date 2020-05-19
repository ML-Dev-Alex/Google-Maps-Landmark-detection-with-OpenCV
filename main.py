# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math
from scipy import ndimage


# Function to show array of images (intermediate results)
def show_images(images, prefix='prefix'):
    for i, img in enumerate(images):
        cv2.imshow(prefix , img)
        cv2.imwrite('results/' + prefix + '_' + str(i) + ".png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_path = "train1/flat/sparseresidential13.tif"

# Read image and preprocess
image = cv2.imread('img.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 3)
show_images([gray], 'gray')

edged = cv2.Canny(gray, 25, 123)
show_images([edged], 'canny')

kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edged, kernel)
show_images([dilated], 'dilated')

final = image.copy()
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


hsv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
color = np.array([200, 60, 40])
colorMax = np.array([255, 100, 80])
mask = cv2.inRange(hsv, color, colorMax)
output = cv2.bitwise_and(hsv, hsv, mask=mask)
points = cv2.findNonZero(mask)
avg = np.mean(points, axis=0)

point = ((image.shape[0] / image.shape[0]) * avg[0][0], (image.shape[1] / image.shape[1]) * avg[0][1])

# find the biggest countour (c) by the area
min = 999999999999999999
min_pos = 0
for i, contour in enumerate(contours):
    cur = abs(cv2.pointPolygonTest(contour, point, True))
    print(cur)
    if min > cur and cv2.contourArea(contour) > 250:
        min = cur
        min_pos = i
    elif cv2.contourArea(contour) > 220:
        temp = image.copy()
print(min)
print(min_pos)

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
cv2.drawContours(final, contours[min_pos], -1, (0, 255, 0), 3)

show_images([final], 'final ')
cv2.imshow('final', np.hstack([final, output]))
cv2.waitKey(0)
