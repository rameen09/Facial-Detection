#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:54:48 2025

@author: nawallrizvi
"""


import cv2

# Load Haar cascade XMLs using absolute paths
eye_data = cv2.CascadeClassifier("/Users/nawallrizvi/Downloads/eye and face project/eye.xml")
face_data = cv2.CascadeClassifier("/Users/nawallrizvi/Downloads/eye and face project/haarcascade_frontalface_default.xml")

# Verify that classifiers loaded correctly
print("Eye loaded:", not eye_data.empty())
print("Face loaded:", not face_data.empty())

# Font for on-screen instructions
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize webcam (0 = default camera)
webcam = cv2.VideoCapture(0)

# Toggle flags for drawing detections
show_eyes = False
show_faces = False

# Display instructions in terminal
print("\n Controls:")
print("    Press 'e' to toggle eye detection")
print("    Press 'f' to toggle face detection")
print("    Press 'n' to clear all overlays")
print("    Press 'q' or ESC to quit\n")

# Main loop
while True:
    frameRead, frame = webcam.read()
    if not frameRead:
        print("Could not read from webcam")
        break

    # Convert to grayscale
    grayScale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features
    eyes = eye_data.detectMultiScale(grayScale_frame, scaleFactor=1.1, minNeighbors=10)
    faces = face_data.detectMultiScale(grayScale_frame, scaleFactor=1.1, minNeighbors=5)

    # Display on-screen instructions
    cv2.putText(frame, "Press E (eye) | F (face) | N (none) | Q (quit)",
                (10, 30), font, 0.6, (255, 255, 255), 2)

    # Draw rectangles for faces
    if show_faces:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw circles for eyes
    if show_eyes:
        for (x, y, w, h) in eyes:
            centerx = x + w // 2
            centery = y + w // 2
            radius = w // 2
            cv2.circle(frame, (centerx, centery), radius, (104, 235, 52), 2)

    # Show the updated frame
    cv2.imshow("Eye and Face Detection with Webcam :)", frame)

    # Key handler
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:  # ESC or 'q'
        print(" Exiting...")
        break
    elif key == ord('e'):
        show_eyes = not show_eyes
        print(" Eye detection:", "ON" if show_eyes else "OFF")
    elif key == ord('f'):
        show_faces = not show_faces
        print(" Face detection:", "ON" if show_faces else "OFF")
    elif key == ord('n'):
        show_eyes = False
        show_faces = False
        print(" All overlays cleared")

# Release camera and close windows
webcam.release()
cv2.destroyAllWindows()