import cv2
import pyvolume
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# detector = HandDetector(maxHands=1, detectionCon=0.8)
detector = HandDetector(detectionCon=0.8)

volScaleMin = 0
volScaleMax = 100

fingerLengthMax = 150


def calculateNewVolumeValue(fingerLengthValue):
    return fingerLengthValue / fingerLengthMax


while cap.isOpened():
    success, img = cap.read()
    # img = cv2.flip(img, 1)

    if success:
        # hands = detector.findHands(img)
        hands, img = detector.findHands(img)
        fingerImg = cv2.imread('images/zero.jpg')

        if hands:
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            fingers1 = detector.fingersUp(hand1)
            count = fingers1.count(1)
            # print(f'H1 = {fingers1.count(1)} \n')  # Print the count of fingers that are up
            print(f"Current count: {count}")

            if count == 1:
                fingerImg = cv2.imread('images/one.jpg')
            if count == 2:
                fingerImg = cv2.imread('images/two.jpeg')
            if count == 3:
                fingerImg = cv2.imread('images/three.jpg')
            if count == 4:
                fingerImg = cv2.imread('images/four.jpg')
            if count == 5:
                fingerImg = cv2.imread('images/five.jpg')
            length, info, img = detector.findDistance(lmList1[4][0:2], lmList1[8][0:2], img, color=(255, 0, 255),
                                                      scale=10)

            print(f"Vol percentage: {calculateNewVolumeValue(length)}")
            pyvolume.custom(percent=calculateNewVolumeValue(length))

        # fingerImg = cv2.resize(fingerImg, (220, 280))
        # img[50:330, 20:240] = fingerImg
        cv2.imshow("My Camera Window", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# cap.release()
# cv2.destroyAllWindows()
