#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string
import cv2
from cv2 import imread
from keras.models import load_model
import time
from processing import square_pad, preprocess
import numpy as np
import copy
import math

"""
LIVE DEMO
This script loads a pre-trained CNN model and classifies American Sign Language
finger spelling in real-time based on filtered camera data
Signum: Software Design SP18 Final Project
Isaac Vandor, Utsav Gupta, Diego Berny
"""

# ====== Set values for filtering and define finger finding ======
# =======================================================
capture_region_x=0.55  # roi x start point
capture_region_y=0.9  # roi y start point
threshold = 65  #  Starting threshold value
blur_value = 9  # GaussianBlur parameter
background_threshold = 0

isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # In case you wanna use virtual keyboard

""" Calculates convexity defects (i.e. fingers) based image and some math"""
def findFingers(result,drawing):  # -> finished bool, count: finger count
#  convexity defect
    result = cv2.approxPolyDP(result,0.01*cv2.arcLength(result,True),True)
    hull = cv2.convexHull(result, returnPoints=False)
    if len(hull) > 2:
    #if(1):
        defects = cv2.convexityDefects(result, hull)
        if type(defects) != type(None):
            count = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(result[s][0])
                end = tuple(result[e][0])
                far = tuple(result[f][0])
                '''
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                count += 1
                '''
                dist = cv2.pointPolygonTest(result,center,True)
            #cv2.line(img,start,end,[0,255,0],2)
            #cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, count
    return False, 0

# ====== Create model for real-time classification ======
# =======================================================
model = load_model('models/my_model.h5')

# Dictionary to convert numerical classes to alphabet
label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}

# ====================== Video Capture Loop =============
# =======================================================
camera_input = int(input('Enter camera number: ')) #Allows you to set any camera you want
video_capture = cv2.VideoCapture(camera_input) # start video capture
video_capture.set(10,200) # set dimensions for video capture
print("Press b to capture background & begin detection or r to reset")

# Start capturing time for FPS analysis
fps = 0
start = time.time()

while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    fps += 1 # Count frames
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(capture_region_x * frame.shape[1]), 0),
                 (frame.shape[1], int(capture_region_y * frame.shape[0])), (255, 0, 0), 2) #Create ROI

# ====================== Video Filtering ================
# =======================================================
    #  Remove Background
    if isBgCaptured == 1:  # Only runs once background is captured
        img = frame
        bgModel.apply(img)
        img = img[0:int(capture_region_y * frame.shape[0]),
                    int(capture_region_x * frame.shape[1]):frame.shape[1]]  # clip the ROI
        #cv2.imshow('mask', img) #show mask image

        # Convert image to grayscale and then binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
        #cv2.imshow('blurred', blur) # show blur image

        '''
        Tries to adaptively change threshold based on lighting conditions
        A background pixel in the center top of the image is sampled to determine
        its intensity.
        '''
        img_width, img_height = np.shape(img)[:2]
        background_level = gray[int(img_height/100)][int(img_width/2)]
        threshold_level = background_threshold + background_level

        # thresholding using Otsu's threshold to get output images
        ret, thresh = cv2.threshold(blur, threshold_level, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh = cv2.dilate(thresh, None, iterations=2)
        #cv2.imshow('binary', thresh) # show binary image

# ====================== Hand Detection =================
# =======================================================
        # get the coutours
        new_threshold = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(new_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(img.shape, np.uint8)

        maxArea = -1

        for i in range(len(contours)):  # find the biggest contour by area
            tmp = contours[i]
            area = cv2.contourArea(tmp)
            if area > maxArea:
                max_area = area
                ci = i

            result = contours[ci]

            # Transformations on contours to find the center of the contour and draw it onscreen
            hull = cv2.convexHull(result)
            moments = cv2.moments(result)
            if moments['m00']!=0:
                        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                        cy = int(moments['m01']/moments['m00']) # cy = M01/M00

            center = (cx, cy)
            #cv2.circle(img,center,5,[0,0,255],2)
            cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            # Call findFingers function to determine finger location
            isFinish,count = findFingers(result,drawing)
            if triggerSwitch is True:
                if isFinish is True and count <= 2:
                    print (count)

        cv2.imshow('Output', drawing)

        # Crop + process captured frame
        #hand = frame[83:650, 314:764] #decided to use better frame detection above
        #hand = square_pad(drawing) #no longer necessary

        hand = preprocess(drawing)

# ============ Model Prediction/Analysis ================
# =======================================================
    # Make prediction
        my_predict = model.predict(hand, batch_size=None, verbose=0, steps=1)
        #print(my_predict) #uncomment for direct model output

        # Predict most likely letter
        top_prd = np.argmax(my_predict)

        # Only display predictions with probabilities greater than 0.05
        if np.max(my_predict) >= 0.05:

            prediction_result = label_dict[top_prd] # convert predictions to alphabet
            preds_list = np.argsort(my_predict)[0] # sort list of predictions
            pred_2 = label_dict[preds_list[-2]] # display 2nd most likely result
            pred_3 = label_dict[preds_list[-3]] # display 3rd most likely result
            #print(preds_list) #uncomment this line to see output
            # set width and height of frame for text annotation
            width = int(video_capture.get(3) + 0.25)
            height = int(video_capture.get(4) + 0.25)

# =========== Display Results ===========================
# =======================================================
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


    # Keyboard Operations to control program
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2()
        isBgCaptured = 1
        print( 'Background Captured: Press 'r' to reset if detection not working')
        print('Blue text is top prediction result; Red and Green are 2nd and 3rd results')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('Background has been reset')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
    elif k == ord('p'):
        height, width = img.shape[:2]
        res = cv2.resize(img,(200, 200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('Data/OutputData/output.jpg',drawing)
        print("Image printed to file")

# Calculate frames per second
end = time.time()
FPS = fps/(end-start)
print("[INFO] approx. FPS: {:.2f}".format(FPS))

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
