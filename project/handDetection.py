# https://gogul.dev/software/hand-gesture-recognition-p1
import cv2
import mediapipe as mediapipe
import imutils
from numberDetection import count
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from handGesture_media import GesturePredict, handProcess

model = load_model('hand_gesture')

# Load class names
file = open('handGesture_names', 'r')
Gestures = file.read().split('\n')
file.close()
print(Gestures)

mediapipeHand = mediapipe.solutions.hands
hand = mediapipeHand.Hands(max_num_hands=1, min_detection_confidence=0.7)
mediaPipeDraw = mediapipe.solutions.drawing_utils


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
    thresholded = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thresholded = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dsta

    (contour, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.CHAIN_APPROX_SIMPLE

    if(len(contour)==0):
        return
    else:
        segmented = max(contour, key= lambda x: cv2.contourArea(x))
        return (thresholded, segmented)

if __name__ == "__main__":
    lamb = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    ccc = 0

    while(True):
        res = 0
        ccc = ccc + 1
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)
        x, y, c = frame.shape

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]
        name = "roi"+str(ccc)+".jpg"
        cv2.imshow("ROI", roi)
        # roi.imwrite(name, array)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (35, 35), 0)
        if(num_frames < 30):
            run_avg(gray, 0.5)
        else:
            bg_average = np.average(bg, axis=0)
            lamb = 75 #black
            if(np.average(bg_average, axis=0)>128):
                lamb = 250
            print(lamb)
            hand = segment(gray, lamb)
            if hand is not None:
                (thresholded, segmented) = hand

                res = count(thresholded, segmented)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imwrite("newimage.png", thresholded)
                cv2.imshow("Thesholded", thresholded)

        processed_result = handProcess(frame)
        gesture = ''
        positions = []
        if processed_result.multi_hand_landmarks:
            positions = []
            for handslms in processed_result.multi_hand_landmarks:
                for mark in handslms.landmark:
                    posX = int(mark.x * x)
                    posY = int(mark.y * y)
                    positions.append([posX, posY])
                gesture = GesturePredict(frame, handslms, positions)
        cv2.putText(clone, gesture, (10, 50), cv2.FONT_HERSHEY_TRIPLEX,
                   1, (50, 52, 168), 2, cv2.LINE_AA)
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1

        cv2.putText(clone, str(res), (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA, False)

        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()
