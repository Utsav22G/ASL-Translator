#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LIVE DEMO
This script loads a pre-trained model and classifies American Sign Language
finger spelling frame-by-frame in real-time
"""

import string
import cv2
from cv2 import imread
from keras.models import load_model
import time
from processing import square_pad, preprocess
import numpy as np

''' Initial Parameters'''
capture_region_x=0.5  # roi x start point
capture_region_y=0.8  # roi y start point
threshold = 70  #  Starting threshold value
blur_value = 5  # GaussianBlur parameter
background_threshold = 60

# ====== Create model for real-time classification ======
# =======================================================
model = load_model('my_model.h5')

# Dictionary to convert numerical classes to alphabet
label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}
# ====================== Live loop ======================
# =======================================================
camera_input = int(input('Enter camera number: '))
video_capture = cv2.VideoCapture(camera_input)
video_capture.set(10,200)

fps = 0
start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    fps += 1
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(capture_region_x * frame.shape[1]), 0),
                 (frame.shape[1], int(capture_region_y * frame.shape[0])), (255, 0, 0), 2)
    '''
    # Draw rectangle around face
    x = 313
    y = 82
    w = 451
    h = 568
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
    '''

    # Crop + process captured frame
    #hand = frame[83:650, 314:764]
    hand = square_pad(frame)

    # Convert image to grayscale and then binary
    #blur = cv2.GaussianBlur(hand, (blur_value, blur_value), 0)
    #cv2.imshow('blurred', blur) # show blur image
    hand = preprocess(hand)

    # Make prediction
    my_predict = model.predict(hand,
                                  batch_size=500,
                                  verbose=0)

    # Predict letter
    top_prd = np.argmax(my_predict)

    # Only display predictions with probabilities greater than 0.5
    if np.max(my_predict) >= 0.50:

        prediction_result = label_dict[top_prd]
        preds_list = np.argsort(my_predict)[0]
        pred_2 = label_dict[preds_list[-2]]
        pred_3 = label_dict[preds_list[-3]]
        print(preds_list)
        width = int(video_capture.get(3) + 0.25)
        height = int(video_capture.get(4) + 0.25)

        # Annotate image with most probable prediction
        cv2.putText(frame, text=prediction_result,
                    org=(width // 2 + 50, height // 2 + 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=10, color=(255, 255, 0),
                    thickness=15, lineType=cv2.LINE_AA)
        # Annotate image with second most probable prediction (displayed on bottom left)
        cv2.putText(frame, text=pred_2,
                    org=(width // 3 + 100, height // 1 + 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=6, color=(0, 255, 0),
                    thickness=6, lineType=cv2.LINE_AA)
        # Annotate image with third probable prediction (displayed on bottom right)
        cv2.putText(frame, text=pred_3,
                    org=(width // 2 + 120, height // 1 + 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=6, color=(0, 0, 255),
                    thickness=6, lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)


    # Press 'q' to exit live loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Calculate frames per second
end = time.time()
FPS = fps/(end-start)
print("[INFO] approx. FPS: {:.2f}".format(FPS))

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
