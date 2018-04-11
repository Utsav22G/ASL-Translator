import cv2
import numpy as np
import copy
import math

# parameters
capture_region_x=0.5  # roi x start point
capture_region_y=0.8  # roi y start point
threshold = 70  #  threshold
blur_value = 5  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # In case you wanna use virtual keyboard


def printThreshold(thr):
    print("! Threshold changed to: "+str(thr))


def removeBG(videostream):
    fgmask = bgModel.apply(videostream)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=5)
    result = cv2.bitwise_and(videostream, videostream, mask=fgmask)
    return result


def findFingers(result,drawing):  # -> finished bool, count: finger count
    #  convexity defect
    result = cv2.approxPolyDP(result,0.01*cv2.arcLength(result,True),True)
    hull = cv2.convexHull(result, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(result, hull)
        if type(defects) != type(None):
            count = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(result[s][0])
                end = tuple(result[e][0])
                far = tuple(result[f][0])
                #a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                #b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                #c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                #angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                #if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                #    count += 1
                dist = cv2.pointPolygonTest(result,center,True)
                cv2.line(img,start,end,[0,255,0],2)
                cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, count
    return False, 0

# Do camera operations
camera_input = int(input('Enter camera number: '))
camera = cv2.VideoCapture(camera_input)
camera.set(10,200)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
print("Press b to capture background & begin detection or r to reset")


while camera.isOpened():
    ret, videostream = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    videostream = cv2.bilateralFilter(videostream, 5, 50, 100)  # smoothing filter
    videostream = cv2.flip(videostream, 1)  # flip the videostream horizontally
    cv2.rectangle(videostream, (int(capture_region_x * videostream.shape[1]), 0),
                 (videostream.shape[1], int(capture_region_y * videostream.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', videostream)

    #  Main operation
    if isBgCaptured == 1:  # Only runs once background is captured
        img = videostream
        img = removeBG(videostream)
        #img = img[0:int(capture_region_y * videostream.shape[0]),
        #            int(capture_region_x * videostream.shape[1]):videostream.shape[1]]  # clip the ROI
        #cv2.imshow('mask', img) #show mask image

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
        cv2.imshow('blurred', blur) # show blur image
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('binary', thresh) # show binary image


        # get the coutours
        new_threshold = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(new_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maxArea = -1

        for i in range(len(contours)):  # find the biggest contour by area
            tmp = contours[i]
            area = cv2.contourArea(tmp)
            if area > maxArea:
                max_area = area
                ci = i

            result = contours[ci]
            hull = cv2.convexHull(result)
            drawing = np.zeros(img.shape, np.uint8)
            moments = cv2.moments(result)
            if moments['m00']!=0:
                        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                        cy = int(moments['m01']/moments['m00']) # cy = M01/M00

            center = (cx, cy)
            cv2.circle(img,center,5,[0,0,255],2)
            cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinish,count = findFingers(result,drawing)
            if triggerSwitch is True:
                if isFinish is True and count <= 2:
                    print (count)


        cv2.imshow('Output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( 'Background Captured: Press 'r' to reset if detection not working')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('Background has been reset')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
