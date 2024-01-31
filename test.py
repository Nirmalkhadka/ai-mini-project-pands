import cv2
from cvzone.HandTrackingModule import HandDetector
#  for sending our image to model
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf

# Initialize the video capture, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants and variables for image processing
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["Hi", "I Love You", "Yes", "No", "Hungry", "Father"]

# Main loop for video processing
while True:
    # Read a frame from the video feed
    success, img = cap.read()
    imgOutput = img.copy()  # Create a copy of the original image for displaying results

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    # Process each detected hand
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image with predefined dimensions
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region around the detected hand
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize and reshape the cropped image to fit the white image
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            # Resize based on height
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # draw flase means don't draw prediction in image white
            print(prediction, index)
        else:
            # Resize based on width
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display rectangles and text around the detected hand and its classification
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED) # for box in text
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2) # color purple
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)# for rectangle around hand

        # Display cropped and processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the annotated original image
    cv2.imshow("Image", imgOutput)

    # Wait for a key event to exit the program
    cv2.waitKey(1)
