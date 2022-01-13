import numpy as np
import cv2
import keyboard
print('Press any key for next frame')
print('Camera 0 Alerts')
video_capture_0 = cv2.VideoCapture('Alertvideo0.avi')
while True:
    ret0, frame0 = video_capture_0.read()
    
    # Display the resulting frame
    if (ret0):
        cv2.imshow('Cam_0', frame0)
    else:
        print('Alert Camera 0 alerts processed or too less')
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cv2.waitKey(0)
video_capture_0 = cv2.VideoCapture('Alertvideo1.avi')
print('Switched to Camera 1 Alerts')
while True:
    ret0, frame0 = video_capture_0.read()
    
    # Display the resulting frame
    if (ret0):
        cv2.imshow('Cam_0', frame0)
    else:
        print('Alert Camera 1 alerts processed or too less')
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cv2.waitKey(0)
video_capture_0 = cv2.VideoCapture('Alertvideo2.avi')
print('Switched to Camera 2 Alerts')
while True:
    ret0, frame0 = video_capture_0.read()
    
    # Display the resulting frame
    if (ret0):
        cv2.imshow('Cam_0', frame0)
    else:
        print('Alert Camera 2 alerts processed or too less')
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cv2.waitKey(0)
# When everything is done, release the capture
cv2.destroyAllWindows()