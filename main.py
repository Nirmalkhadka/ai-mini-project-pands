import cv2  # Import OpenCV library
from cvzone.HandTrackingModule import HandDetector  # Import HandDetector from cvzone library
import numpy as np
import math
import time

# Open video capture (camera index 0)
cap = cv2.VideoCapture(0)

# Initialize HandDetector with a maximum of 1 hand to be detected
detector = HandDetector(maxHands=1)

# Offset for cropping around the detected hand
offset = 20

# Size of the square image to be saved
imgSize = 300

# Folder to save the captured images
folder = "Data/F"

# Counter to keep track of the number of captured images
counter = 0

# Main loop to continuously capture and process video frames
while True:
    # Read a frame from the video feed
    success, img = cap.read()

    # Detect hands in the frame using HandDetector
    hands, img = detector.findHands(img)

    # If hands are detected
    if hands:
        hand = hands[0]  # we only have one hand so we write it like that
        x, y, w, h = hand['bbox']   # it will give all the value of width and height x,y

        # Create a white image of size imgSize x imgSize
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # 0 to 255 is range of the colour so we multiply by 255

        # Crop the region around the detected hand with an offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Determine aspect ratio and resize the cropped image accordingly
        aspectRatio = h / w
        if aspectRatio > 1:  # if it is above 1 this means height is greater
            # this part give solution to height so height adjust according to our hand in camera
            k = imgSize / h  # we are streching the height
            wCal = math.ceil(k * w)  # it gives calculated width and ceil means always give value more then roundof
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)  # this will center the image
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # this part give solution to width so width adjust according to our hand in camera
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display the cropped and resized images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original frame with hand detection
    cv2.imshow("Image", img)

    # Wait for a key press with a delay of 1 millisecond
    key = cv2.waitKey(1) & 0xFF

    # If the 's' key is pressed, save the resized image to the folder
    if key == ord("s"):
        counter += 1
        # Save the image with a filename containing the current timestamp
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
