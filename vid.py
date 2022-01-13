import numpy as np
import cv2
import keyboard


video_capture_0 = cv2.VideoCapture('http://192.168.10.5:8081/video')
# video_capture_1 = cv2.VideoCapture(0)

cam0_show = True


out0 = cv2.VideoWriter('video0.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_0.get(3)),int(video_capture_0.get(4))))


print("Press A to toggle Camera 1")


while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()

    if keyboard.is_pressed('a'):
        cam0_show = not cam0_show
        if cam0_show == False:
            cv2.destroyWindow("Cam_0")
    

 

    # Display the resulting frame
    out0.write(frame0) 
    if (ret0) and (cam0_show):
        cv2.imshow('Cam_0', frame0)
        
        

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
cv2.destroyAllWindows()