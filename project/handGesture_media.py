import cv2 
import numpy as np
import mediapipe as mediapipe
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading the hand gesture recognize model
model = load_model('hand_gesture')

# Load class names
file = open('handGesture_names', 'r')
Gestures = file.read().split('\n')
file.close()
print(Gestures)

mediapipeHand = mediapipe.solutions.hands
hand = mediapipeHand.Hands(max_num_hands=1, min_detection_confidence=0.7)
mediaPipeDraw = mediapipe.solutions.drawing_utils

def handProcess(VideoFrame):
     VideoFrame = cv2.flip(VideoFrame, 1)
     framergb = cv2.cvtColor(VideoFrame, cv2.COLOR_BGR2RGB)
     processed_result = hand.process(framergb)
     return processed_result

def GesturePredict(videoframe, x, positions):
    mediaPipeDraw.draw_landmarks(videoframe, x, mediapipeHand.HAND_CONNECTIONS)
    # Predict gesture
    prediction = model.predict([positions])
    print(prediction)
    classID = np.argmax(prediction)
    gesture = Gestures[classID]
    print("Prediction=", prediction,"Resultant Gesture=", gesture)
    f = open("predicted.txt", "a")
    f.write(gesture)
    f.write("\n")
    f.close()
    return gesture