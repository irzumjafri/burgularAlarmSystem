import numpy as np
import cv2
import keyboard

video_capture_0 = cv2.VideoCapture('http://192.168.10.23:8080/video')
video_capture_1 = cv2.VideoCapture('http://192.168.10.28:8081/video')
video_capture_2 = cv2.VideoCapture('http://192.168.10.11:8081/video')
# video_capture_1 = cv2.VideoCapture(0)

cam0_show = True
cam1_show = True
cam2_show = True

out0 = cv2.VideoWriter('video0.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_0.get(3)),int(video_capture_0.get(4))))
out1 = cv2.VideoWriter('video1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_1.get(3)),int(video_capture_1.get(4))))
out2 = cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_1.get(3)),int(video_capture_1.get(4))))

print("Press A to toggle Camera 1")
print("Press S to toggle Camera 2")
print("Press D to toggle Camera 3")

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    if keyboard.is_pressed('a'):
        cam0_show = not cam0_show
        if cam0_show == False:
            cv2.destroyWindow("Cam_0")
    
    if keyboard.is_pressed('s'):
        cam1_show = not cam1_show
        if cam1_show == False:
            cv2.destroyWindow("Cam_1")
    
    if keyboard.is_pressed('d'):
        cam2_show = not cam2_show
        if cam2_show == False:
            cv2.destroyWindow("Cam_2")

    # Display the resulting frame
    out0.write(frame0) 
    if (ret0) and (cam0_show):
        cv2.imshow('Cam_0', frame0)
        

    # Display the resulting frame
    out1.write(frame1) 
    if (ret1) and (cam1_show):
        cv2.imshow('Cam_1', frame1)
        

    # Display the resulting frame
    out2.write(frame2) 
    if (ret2) and (cam2_show):
        cv2.imshow('Cam_2', frame2)
        

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
video_capture_2.release()
cv2.destroyAllWindows()