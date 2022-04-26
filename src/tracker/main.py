import cv2
import sys
import numpy as np

SCREEN_STD = "std"
SCREEN_HSV = "hsv"
ESC = 27

camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
vid_cap = cv2.VideoCapture(camera_idx)

light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)


def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, " ", y)


def thresh_callback(im, drawing):
    #threshold = val
    
    #canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    
    
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
    #drawing = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    
    for i, c in enumerate(contours):
        # contour
        #cv2.drawContours(drawing, contours, i, (0, 0, 255))
        # ellipse
        # if c.shape[0] > 5:
        #     cv2.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(drawing, [box], 0, (0, 0, 255))
    
    
    cv2.imshow('Contours', drawing)


def track():
    blob_param = cv2.SimpleBlobDetector_Params()
    blob_param.filterByArea = True
    blob_param.minArea = 1
    #blob_param.maxArea = 100
    blob_param.filterByColor = True
    blob_param.blobColor = 255
    blob_param.filterByConvexity = False
    blob_param.filterByInertia = False
    blob_param.minDistBetweenBlobs = 1
    blob_param.filterByCircularity = False
    # blob_param.filterByArea = True
    # blob_param.minArea = 1000

    blob_detector = cv2.SimpleBlobDetector_create(blob_param)

    kernel = np.ones((5,5), np.uint8)

    while True:
        ret, frame = vid_cap.read()

        cv2.imshow(SCREEN_STD, frame)
        cv2.setMouseCallback(SCREEN_STD, mouse_click)


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, light_orange, dark_orange)
        
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        cv2.imshow(SCREEN_HSV, mask)

        # keypoints = blob_detector.detect(mask)
        # blobs_mask = cv2.drawKeypoints(frame, keypoints, np.zeros((1, 1)), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("", blobs_mask)


        thresh_callback(mask, frame)

        if cv2.waitKey(1) == ESC:
            break


if __name__ == "__main__":
    track()
    cv2.destroyAllWindows()
    vid_cap.release()
