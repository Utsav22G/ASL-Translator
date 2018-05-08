import cv2
from cv2 import imread
from keras.models import load_model
from PIL import Image
import numpy as np

"""
Live Predict
This script loads a pre-trained CNN model and classifies American Sign Language
finger spelling from from live camera input with no filtering on the camera frame
Signum: Software Design SP18 Final Project
Isaac Vandor, Utsav Gupta, Diego Berny
"""

#model = load(trained_model='model.h5')
model = load_model('models/model.h5')
#result = predict(trained_model=model, test_image=frame)
#'Data/OutputData/scaledoutput.jpg'
alphabets = {"A": 0, "B":1, "C": 2, "D":3, "E": 4, "F": 5,
            "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
            "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "S": 17,
            "T": 18, "U": 19, "V": 20, "W": 21, "X": 22, "Y": 23}

#alphabet = find_alphabet(letter_list=result, letter_dict=alphabets)
#print("The alphabet is: " + alphabet)

capture_region_x=0.55  # roi x start point
capture_region_y=0.9  # roi y start point
size = 200

# ====================== Live loop ======================
# =======================================================
camera_input = int(input('Enter camera number: '))
video_capture = cv2.VideoCapture(camera_input)
video_capture.set(10,200)
#print("Press b to capture background & begin detection or r to reset")

while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(capture_region_x * frame.shape[1]), 0),
                 (frame.shape[1], int(capture_region_y * frame.shape[0])), (255, 0, 0), 2)

    frame = cv2.resize(frame, (size, size))
    img = frame
    img = img.astype(np.float32)/255.0      # convert to float32
    img = img[:,:,::-1]         # convert from RGB to BGR

    result = model.predict(np.expand_dims(frame, axis=0))[0]

    letter_list=result
    letter_dict = alphabets
    idx = letter_list.argmax(axis=0)    # find the index of the biggest argument

    # look for the key corresponding to the biggest argument
    decoded = [key for key, value in letter_dict.items() if value == idx]
    alphabet = decoded[0]

    width = int(video_capture.get(3) + 0.25)
    height = int(video_capture.get(4) + 0.25)

    # Annotate image with most probable prediction
    cv2.putText(frame, text=alphabet,
                org=(width // 2 + 50, height // 2 + 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=10, color=(255, 255, 0),
                thickness=15, lineType=cv2.LINE_AA)

    #result = predict(trained_model=model, test_image=frame)
    #alphabet = find_alphabet(letter_list=result, letter_dict=alphabets)
    print("The alphabet is: " + alphabet)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
# Release the capture
video_capture.release()
cv2.destroyAllWindows()
