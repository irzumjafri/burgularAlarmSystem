import numpy as np
import cv2
import keyboard
from numpy.matrixlib.defmatrix import matrix


video_capture_0 = cv2.VideoCapture('testingvideo0.avi')
video_capture_1 = cv2.VideoCapture('testingvideo1.avi')
video_capture_2 = cv2.VideoCapture('testingvideo2.avi')


# video_capture_0 = cv2.VideoCapture('http://192.168.10.5:8081/video')
# video_capture_1 = cv2.VideoCapture('http://192.168.10.2:8081/video')
# video_capture_2 = cv2.VideoCapture('http://192.168.10.9:8080/video')

cam0_show = True
cam1_show = True
cam2_show = True



# out0 = cv2.VideoWriter('video0.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_0.get(3)),int(video_capture_0.get(4))))
# out0_top = cv2.VideoWriter('video0_top.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_0.get(3)),int(video_capture_0.get(4))))
# out1 = cv2.VideoWriter('video1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_1.get(3)),int(video_capture_1.get(4))))
# out1_top = cv2.VideoWriter('video1_top.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_1.get(3)),int(video_capture_1.get(4))))
# out2 = cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_1.get(3)),int(video_capture_1.get(4))))
# out2_top = cv2.VideoWriter('video2_top.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture_1.get(3)),int(video_capture_1.get(4))))


print("Press A to toggle Camera 1 CAM VIEW")
print("Press S to toggle Camera 2 CAM VIEW")
print("Press D to toggle Camera 3 CAM VIEW")

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    if keyboard.is_pressed('a'):
        cam0_show = not cam0_show
        if cam0_show == False:
            cv2.destroyWindow("Cam_0")
    im_src = frame0

    pts_src = np.array([[93, 254], [412, 260], [434, 324], [391, 327], [401, 372], [56, 363], [2, 372]])
    pts_dst = np.array([[49, 373], [51, 36], [147, 35], [235, 71], [290, 73], [291, 296], [244, 371]])
    # pts_src = np.array([[60,360],[157,163],[309,180],[415,262],[433,325],[493,301],[1,438]])
    # pts_dst = np.array([[290,297],[48,297],[48,148],[48,34],[147,34],[290,34],[420,298]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(im_src, h, (640,480))

    cv2.imshow('output0', im_out)

    
    if keyboard.is_pressed('s'):
        cam1_show = not cam1_show
        if cam1_show == False:
            cv2.destroyWindow("Cam_1")

    im_src1 = frame1
    pts_src1 = np.array([[281, 409], [349, 399], [287, 373], [13, 457], [101, 476],[15, 459], [286, 374], [352, 399], [315, 426], [188, 478]])
    pts_dst1 = np.array([[137, 110], [144, 36], [50, 35], [49, 350], [136, 297],[49, 354], [51, 36], [148, 36], [229, 110], [235, 294]])
    # pts_src1 = np.array([[15, 459], [286, 374], [352, 399], [315, 426], [188, 478]])
    # pts_dst1 = np.array([[49, 354], [51, 36], [148, 36], [229, 110], [235, 294]])
    h1, status1 = cv2.findHomography(pts_src1, pts_dst1)
    im_out1 = cv2.warpPerspective(im_src1, h1, (640,480))

    cv2.imshow('output1', im_out1)
    
    
    if keyboard.is_pressed('d'):
        cam2_show = not cam2_show
        if cam2_show == False:
            cv2.destroyWindow("Cam_2")
    im_src2 = frame2
    pts_src2 = np.array([[56, 356],[331, 318], [391, 301], [215, 382], [186, 325], [394,473],[140,329]])
    pts_dst2 = np.array([[50, 36], [234, 72], [289, 73], [140, 109], [122, 150], [211,211],[99,34]])
    # pts_src2 = np.array([[37, 478], [26, 369], [56, 356], [221, 308], [304, 284], [331, 318], [391, 301], [215, 382], [186, 325], [394,473],[140,329]])
    # pts_dst2 = np.array([[48, 229], [49, 149], [50, 36], [145, 35], [233, 34], [234, 72], [289, 73], [136, 109], [122, 150], [211,211],[99,34]])
    # pts_src2 = np.array([[55, 356], [221, 309], [305, 284], [334, 320], [395, 304], [561, 478], [545, 477], [403, 321], [217, 383]])
    # pts_dst2 = np.array([[50, 33], [146, 34], [233, 35], [235, 72], [289, 71], [290, 242], [285, 227], [285, 111], [137, 111]])
    # pts_src2 = np.array([[545, 476], [403, 319], [215, 383], [217, 309], [306, 231], [332, 265], [394, 246]])
    # pts_dst2 = np.array([[286, 269], [286, 110], [142, 110], [142, 35], [233, 35], [236, 72], [289, 70]])
    h2, status2 = cv2.findHomography(pts_src2, pts_dst2)
    im_out2 = cv2.warpPerspective(im_src2, h2, (640,480))

    cv2.imshow('output2', im_out2)

    
    img2 = ((im_out/255)+(im_out1/255)+(im_out2/255))/3
    mask2 = np.where(img2 == im_out2,img2,im_out2)
    mask1 = np.where(img2 == im_out1,img2,im_out1)
    im_out1 = mask1*(im_out1/255)/2
    im_out2 = mask2*(im_out2/255)/2
    img2 = ((im_out/255)+(im_out1/255)+(im_out2/255))

    cv2.imshow('joint', img2)

    # stitch = cv2.Stitcher.create()
    # (status, result) = stitch.stitch([im_out,im_out1, im_out2])
    # if status == cv2.STITCHER_OK:
    #     cv2.imshow('joint', result)
    # else:
    #     print("fail")

    # stitch1 = cv2.Stitcher.create()
    # (status1, result1) = stitch1.stitch([frame2, frame1, frame0])
    # cv2.imshow('jointo', result1)
    

    # Display the resulting frame
    # out0.write(frame0) 
    # out0_top.write(im_out)
    # if (ret0) and (cam0_show):
    #     cv2.imshow('Cam_0', frame0)
        

    # Display the resulting frame
    # out1.write(frame1) 
    # out1_top.write(im_out1)
    # if (ret1) and (cam1_show):
    #     cv2.imshow('Cam_1', frame1)
        

    # # Display the resulting frame
    # out2.write(frame2)
    # out2_top.write(im_out2)
    # if (ret2) and (cam2_show):
    #     cv2.imshow('Cam_2', frame2)
        

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
video_capture_2.release()
cv2.destroyAllWindows()



