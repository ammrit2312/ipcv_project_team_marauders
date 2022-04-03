# https://gogul.dev/software/hand-gesture-recognition-p1
import cv2
import imutils
import numpy as np

bg = None

def run_avg(image, alpha):
    global bg
    
    if bg is None:
        bg = image.copy().astype("float")
        return
    
    cv2.accumulateWeighted(image, bg, alpha)

# dst(x,y)=(1−a).dst(x,y)+a.src(x,y)

def segment(image, lamb = 50):
    global bg
    #taking absolute difference between the image with foreground and background
    diff = cv2.absdiff(bg.astype('uint8'), image)

    #thresholding the image
    thresholded = cv2.threshold(diff, lamb, 255, cv2.THRESH_BINARY)[1]

    (contour, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contour)==0):
        return
    else:
        segmented = max(contour, key=cv2.contourArea)
        return (thresholded, segmented)

if __name__ == "__main__":
    lamb = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if(num_frames < 30):
            run_avg(gray, 0.5)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        num_frames += 1

        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()