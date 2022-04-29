import numpy as np
from sklearn.metrics import pairwise
import cv2

def count(thresholded, segmented):
    convH = cv2.convexHull(segmented)

    extreme_top = tuple(convH[convH[:, :, 1].argmin()][0]) #gets min y value of the contour (top) get that list of coordinates
    extreme_bottom = tuple(convH[convH[:, :, 1].argmax()][0]) #gets max y value of the contour (bottom) get that list of coordinates
    extreme_left   = tuple(convH[convH[:, :, 0].argmin()][0]) #gets min x value of the contour (left) get that list of coordinates
    extreme_right  = tuple(convH[convH[:, :, 0].argmax()][0]) #gets max y value of the contour (right) get that list of coordinates

    centerX = int((extreme_left[0] + extreme_right[0])/2)
    centerY = int((extreme_top[1] + extreme_bottom[1])/2)

    distance = pairwise.euclidean_distances([(centerX, centerY)], Y = [extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    max_distance = distance[distance.argmax()]

    radius = int(0.8 * max_distance)

    circum = (2 * np.pi * radius)
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    cv2.circle(circular_roi, (centerX, centerY), radius, 255, 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    (contour, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0

    for c in contour:
        (x, y, w, h) = cv2.boundingRect(c)

        if ((centerY + (centerY * 0.25)) > (y + h)) and ((circum * 0.25) > c.shape[0]):
            count += 1
    
    return count
    # print(max_distance)