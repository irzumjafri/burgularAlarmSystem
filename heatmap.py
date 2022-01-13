import cv2
import argparse
import numpy as np
import keyboard
from numpy.lib.type_check import imag
import matplotlib.pyplot as plt

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

cam_show = True
# video_out = cv2.VideoWriter("video.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
# video_out1 = cv2.VideoWriter("video1.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
# video_out2 = cv2.VideoWriter("video2.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))

# video_static = cv2.VideoWriter("videostatic.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
# video_animated = cv2.VideoWriter("videoanimated.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))


# print("Press A to toggle Camera 1")
# print("Press S to toggle Camera 2")
# print("Press D to toggle Camera 3")

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
    

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


results1= open("results1.txt","w+")
results2= open("results2.txt","w+")
results3= open("results3.txt","w+")

video_capture_0 = cv2.VideoCapture("testingvideo0.avi")
video_capture_1 = cv2.VideoCapture("testingvideo1.avi")
video_capture_2 = cv2.VideoCapture("testingvideo2.avi")

pts_src = np.array([[93, 254], [412, 260], [434, 324], [391, 327], [401, 372], [56, 363], [2, 372]])
pts_dst = np.array([[49, 373], [51, 36], [147, 35], [235, 71], [290, 73], [291, 296], [244, 371]])
homo, status = cv2.findHomography(pts_src, pts_dst)

pts_src1 = np.array([[281, 409], [349, 399], [287, 373], [13, 457], [101, 476],[15, 459], [286, 374], [352, 399], [315, 426], [188, 478]])
pts_dst1 = np.array([[137, 110], [144, 36], [50, 35], [49, 350], [136, 297],[49, 354], [51, 36], [148, 36], [229, 110], [235, 294]])
homo1, status1 = cv2.findHomography(pts_src1, pts_dst1)

pts_src2 = np.array([[56, 356],[331, 318], [391, 301], [215, 382], [186, 325], [394,473],[140,329]])
pts_dst2 = np.array([[50, 36], [234, 72], [289, 73], [140, 109], [122, 150], [211,211],[99,34]])
homo2, status2 = cv2.findHomography(pts_src2, pts_dst2)

heatmapcords = []
heatmapcords1 = []
heatmapcords2 = []
aheatmapcords = []
aheatmapcords1 = []
aheatmapcords2 = []

ANIMATED_COUNTER = 0




classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net0 = cv2.dnn.readNet(args.weights, args.config)
net1 = cv2.dnn.readNet(args.weights, args.config)
net2 = cv2.dnn.readNet(args.weights, args.config)

while True:

    if ANIMATED_COUNTER == 5: #SET K HERE FOR ANIMATED HEATMAP
        ANIMATED_COUNTER = 0
        aheatmapcords = []
        aheatmapcords1 = []
        aheatmapcords2 = []
        print('ANIMATED HEATMAP CLEANED AFTER 5 FRAMES')

    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    image0 = frame0
    image1 = frame1
    image2 = frame2

    blob0 = cv2.dnn.blobFromImage(image0, 0.00392, (640,480), (0,0,0), True, crop=False)
    blob1 = cv2.dnn.blobFromImage(image1, 0.00392, (640,480), (0,0,0), True, crop=False)
    blob2 = cv2.dnn.blobFromImage(image2, 0.00392, (640,480), (0,0,0), True, crop=False)

    net0.setInput(blob0)
    net1.setInput(blob1)
    net2.setInput(blob2)

    outs0 = net0.forward(get_output_layers(net0))
    outs1 = net1.forward(get_output_layers(net1))
    outs2 = net2.forward(get_output_layers(net2))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold, nms_threshold = 0.5, 0.4
    for out in outs0:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y  = int(detection[0] * image0.shape[1]), int(detection[1] * image0.shape[0])
                w, h = int(detection[2] * image0.shape[1]), int(detection[3] * image0.shape[0])
                x, y = center_x - w / 2, center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        str_out = str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n"  
        results1.write(str_out)
        # draw_bounding_box(image0, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        heatmapcords.append([round(x+(w/2)), round(y+(h/2))])
        aheatmapcords.append([round(x+(w/2)), round(y+(h/2))])
        # image0 = cv2.circle(image0, (round(x+(w/2)), round(y+(h))), 2, (255, 0, 0), -1)



    class_ids = []
    confidences = []
    boxes = []
    conf_threshold, nms_threshold = 0.5, 0.4
    for out in outs1:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y  = int(detection[0] * image1.shape[1]), int(detection[1] * image1.shape[0])
                w, h = int(detection[2] * image1.shape[1]), int(detection[3] * image1.shape[0])
                x, y = center_x - w / 2, center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        str_out = str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n"  
        results2.write(str_out)
        # draw_bounding_box(image1, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        heatmapcords1.append([round(x+(w/2)), round(y+(h))])
        aheatmapcords1.append([round(x+(w/2)), round(y+(h))])        
        # image1 = cv2.circle(image1, (round(x+(w/2)), round(y+(h))), 2, (255, 0, 0), -1)



    class_ids = []
    confidences = []
    boxes = []
    conf_threshold, nms_threshold = 0.5, 0.4
    for out in outs2:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y  = int(detection[0] * image2.shape[1]), int(detection[1] * image2.shape[0])
                w, h = int(detection[2] * image2.shape[1]), int(detection[3] * image2.shape[0])
                x, y = center_x - w / 2, center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        str_out = str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n"
        results3.write(str_out)
        # draw_bounding_box(image2, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        heatmapcords2.append([round(x+(w/2)), round(y+(h))])
        aheatmapcords2.append([round(x+(w/2)), round(y+(h))])
        # image2 = cv2.circle(image2, (round(x+(w/2)), round(y+(h))), 2, (255, 0, 0), -1)


    # if keyboard.is_pressed('a'): 
    #     cam_show = not cam_show
    #     if cam_show == False:
    #         cv2.destroyWindow("Live Cam 1")

    # if (ret0) and (cam_show):
    #     cv2.imshow("Live Cam 1", image0)
    #     cv2.imshow("Live Cam 2", image1)
    #     cv2.imshow("Live Cam 3", image2)

    im_out = cv2.warpPerspective(image0, homo, (640,480))

    im_out1 = cv2.warpPerspective(image1, homo1, (640,480))

    im_out2 = cv2.warpPerspective(image2, homo2, (640,480))

    img2 = ((im_out/255)+(im_out1/255)+(im_out2/255))/3
    mask2 = np.where(img2 == im_out2,img2,im_out2)
    mask1 = np.where(img2 == im_out1,img2,im_out1)
    im_out1 = mask1*(im_out1/255)/2
    im_out2 = mask2*(im_out2/255)/2
    img2 = ((im_out/255)+(im_out1/255)+(im_out2/255))

    cv2.imshow('No Heatmap', img2)



    kheat=21
    gauss=cv2.getGaussianKernel(kheat,np.sqrt(64))
    gauss=gauss*gauss.T
    gauss=(gauss/gauss[int(kheat/2),int(kheat/2)])
    cg=cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),3),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # plt.imshow(cg)
    # plt.show()
    gaussbase = np.zeros((480,640,3)).astype(np.float32)

    for p in heatmapcords:
        b = gaussbase[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:]
        c = cg + b
        gaussbase[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:] = c

    m = np.max(gaussbase,axis=(0,1))+0.0001
    gaussbase = gaussbase/m
    # cv2.imshow('gaussbase', gaussbase)

    kheat=21
    gauss=cv2.getGaussianKernel(kheat,np.sqrt(64))
    gauss=gauss*gauss.T
    gauss=(gauss/gauss[int(kheat/2),int(kheat/2)])
    cg=cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),3),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # plt.imshow(cg)
    # plt.show()
    gaussbase1 = np.zeros((480,640,3)).astype(np.float32)


    for p in heatmapcords1:
        b = gaussbase1[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:]
        c = cg + b
        gaussbase1[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:] = c

    m = np.max(gaussbase1,axis=(0,1))+0.0001
    gaussbase1 = gaussbase1/m
    # cv2.imshow('gaussbase1', gaussbase1)

    kheat=21
    gauss=cv2.getGaussianKernel(kheat,np.sqrt(64))
    gauss=gauss*gauss.T
    gauss=(gauss/gauss[int(kheat/2),int(kheat/2)])
    cg=cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),3),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # plt.imshow(cg)
    # plt.show()
    gaussbase2 = np.zeros((480,640,3)).astype(np.float32)


    for p in heatmapcords2:
        b = gaussbase2[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:]
        c = cg + b
        gaussbase2[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:] = c

    m = np.max(gaussbase2,axis=(0,1))+0.0001
    gaussbase2 = gaussbase2/m

    
    gauss_out = cv2.warpPerspective(gaussbase, homo, (640,480))
    gauss_out1 = cv2.warpPerspective(gaussbase1, homo1, (640,480))
    gauss_out2 = cv2.warpPerspective(gaussbase2, homo2, (640,480))


    
    
    gover = cv2.cvtColor(gauss_out,cv2.COLOR_BGR2GRAY)
    gmask = np.where(gover>0.2,1,0).astype(np.float32)
    # cv2.imshow('gmask',gmask) 
    gover1 = cv2.cvtColor(gauss_out1,cv2.COLOR_BGR2GRAY)
    gmask1 = np.where(gover1>0.2,1,0).astype(np.float32)
    # cv2.imshow('gmask1',gmask1) 
    gover2 = cv2.cvtColor(gauss_out2,cv2.COLOR_BGR2GRAY)
    gmask2 = np.where(gover2>0.2,1,0).astype(np.float32)
    # cv2.imshow('gmask2',gmask2) 

    gauss_outmask = gauss_out*(gmask)[:,:,None]
    gauss_outmask1 = gauss_out1*(gmask1)[:,:,None]
    gauss_outmask2 = gauss_out2*(gmask2)[:,:,None]


    gauss_img2 = img2 + gauss_outmask
    gauss_img2 = gauss_img2 + gauss_outmask1    
    gauss_img2 = gauss_img2 + gauss_outmask2



    cv2.imshow('STATIC HEATMAP', gauss_img2)
    # cv2.imwrite('sheatmap.jpg', gauss_img2)

    # video_static.write(gauss_img2)


    # ANIMATED HEAT

    kheat=21
    gauss=cv2.getGaussianKernel(kheat,np.sqrt(64))
    gauss=gauss*gauss.T
    gauss=(gauss/gauss[int(kheat/2),int(kheat/2)])
    cg=cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),21),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # plt.imshow(cg)
    # plt.show()
    gaussbase = np.zeros((480,640,3)).astype(np.float32)

    for p in aheatmapcords:
        b = gaussbase[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:]
        c = cg + b
        gaussbase[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:] = c

    m = np.max(gaussbase,axis=(0,1))+0.0001
    gaussbase = gaussbase/m
    # cv2.imshow('gaussbase', gaussbase)

    kheat=21
    gauss=cv2.getGaussianKernel(kheat,np.sqrt(64))
    gauss=gauss*gauss.T
    gauss=(gauss/gauss[int(kheat/2),int(kheat/2)])
    cg=cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),21),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # plt.imshow(cg)
    # plt.show()
    gaussbase1 = np.zeros((480,640,3)).astype(np.float32)


    for p in aheatmapcords1:
        b = gaussbase1[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:]
        c = cg + b
        gaussbase1[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:] = c

    m = np.max(gaussbase1,axis=(0,1))+0.0001
    gaussbase1 = gaussbase1/m
    # cv2.imshow('gaussbase1', gaussbase1)

    kheat=21
    gauss=cv2.getGaussianKernel(kheat,np.sqrt(64))
    gauss=gauss*gauss.T
    gauss=(gauss/gauss[int(kheat/2),int(kheat/2)])
    cg=cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),21),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # plt.imshow(cg)
    # plt.show()
    gaussbase2 = np.zeros((480,640,3)).astype(np.float32)


    for p in aheatmapcords2:
        b = gaussbase2[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:]
        c = cg + b
        gaussbase2[p[1]-int(kheat/2):p[1]+int(kheat/2)+1, p[0]-int(kheat/2): p[0]+int(kheat/2)+1,:] = c

    m = np.max(gaussbase2,axis=(0,1))+0.0001
    gaussbase2 = gaussbase2/m

    
    gauss_out = cv2.warpPerspective(gaussbase, homo, (640,480))
    gauss_out1 = cv2.warpPerspective(gaussbase1, homo1, (640,480))
    gauss_out2 = cv2.warpPerspective(gaussbase2, homo2, (640,480))


    
    
    gover = cv2.cvtColor(gauss_out,cv2.COLOR_BGR2GRAY)
    gmask = np.where(gover>0.2,1,0).astype(np.float32)
    # cv2.imshow('gmask',gmask) 
    gover1 = cv2.cvtColor(gauss_out1,cv2.COLOR_BGR2GRAY)
    gmask1 = np.where(gover1>0.2,1,0).astype(np.float32)
    # cv2.imshow('gmask1',gmask1) 
    gover2 = cv2.cvtColor(gauss_out2,cv2.COLOR_BGR2GRAY)
    gmask2 = np.where(gover2>0.2,1,0).astype(np.float32)
    # cv2.imshow('gmask2',gmask2) 

    gauss_outmask = gauss_out*(gmask)[:,:,None]
    gauss_outmask1 = gauss_out1*(gmask1)[:,:,None]
    gauss_outmask2 = gauss_out2*(gmask2)[:,:,None]


    gauss_img2 = img2 + gauss_outmask
    gauss_img2 = gauss_img2 + gauss_outmask1    
    gauss_img2 = gauss_img2 + gauss_outmask2



    cv2.imshow('ANIMATED HEATMAP', gauss_img2)
    ANIMATED_COUNTER +=1

    # video_animated.write(gauss_img2)

    # cv2.imshow('No Heatmap', img2)
    # cv2.imwrite('joint.jpg', img2)
    # img2 = cv2.imread('joint.jpg')

    # cv2.imshow('joint', im_out)
    # cv2.imshow('joint1', im_out1)
    # cv2.imshow('joint2', im_out2)

    # video_out.write(image0)
    # video_out1.write(image1)
    # video_out2.write(image2)

    
    cv2.waitKey(1)            
        # release resources
cv2.destroyAllWindows()
# video_out.release()