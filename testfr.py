import cv2
import face_recognition
import os

capture = cv2.VideoCapture(0)


while(True):
    #variables
    ret, frame = capture.read()
    keypressed = cv2.waitKey(1) & 0xFF
    # gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Capture Image',frame)
    
    
    if keypressed==ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows