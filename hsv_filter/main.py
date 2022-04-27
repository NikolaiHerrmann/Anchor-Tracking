
import cv2
import sys
import numpy as np

SCREEN_MAIN = "Thresholds"


def nothing(x):
    pass


cv2.namedWindow(SCREEN_MAIN)

cv2.createTrackbar('HMin', SCREEN_MAIN, 0, 179, nothing)
cv2.createTrackbar('SMin', SCREEN_MAIN, 0, 255, nothing)
cv2.createTrackbar('VMin', SCREEN_MAIN, 0, 255, nothing)
cv2.createTrackbar('HMax', SCREEN_MAIN, 0, 179, nothing)
cv2.createTrackbar('SMax', SCREEN_MAIN, 0, 255, nothing)
cv2.createTrackbar('VMax', SCREEN_MAIN, 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', SCREEN_MAIN, 179)
cv2.setTrackbarPos('SMax', SCREEN_MAIN, 255)
cv2.setTrackbarPos('VMax', SCREEN_MAIN, 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
vid_cap = cv2.VideoCapture(camera_idx)
waitTime = 33

while True:

    _, frame = vid_cap.read()

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', SCREEN_MAIN)
    sMin = cv2.getTrackbarPos('SMin', SCREEN_MAIN)
    vMin = cv2.getTrackbarPos('VMin', SCREEN_MAIN)

    cv2.setTrackbarMax('HMin', SCREEN_MAIN, 12)

    hMax = cv2.getTrackbarPos('HMax', SCREEN_MAIN)
    sMax = cv2.getTrackbarPos('SMax', SCREEN_MAIN)
    vMax = cv2.getTrackbarPos('VMax', SCREEN_MAIN)

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    # Display output image
    cv2.imshow(SCREEN_MAIN, output)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
vid_cap.release()
