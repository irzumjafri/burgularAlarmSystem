import cv2
import argparse
import numpy as np
import keyboard
from numpy.lib.type_check import imag

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
video_out = cv2.VideoWriter("video.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
video_out1 = cv2.VideoWriter("video1.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
video_out2 = cv2.VideoWriter("video2.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))

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


results= open("results1.txt","w+")
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



classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net0 = cv2.dnn.readNet(args.weights, args.config)
net1 = cv2.dnn.readNet(args.weights, args.config)
net2 = cv2.dnn.readNet(args.weights, args.config)

while True:
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
        results.write(str_out)
        # draw_bounding_box(image0, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        image0 = cv2.circle(image0, (round(x+(w/2)), round(y+(h))), 5, (255, 0, 0), -1)



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
        results.write(str_out)
        # draw_bounding_box(image1, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        image1 = cv2.circle(image1, (round(x+(w/2)), round(y+(h))), 5, (255, 0, 0), -1)



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
        results.write(str_out)
        # draw_bounding_box(image2, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        image2 = cv2.circle(image2, (round(x+(w/2)), round(y+(h))), 5, (255, 0, 0), -1)


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

    cv2.imshow('joint', im_out)
    cv2.imshow('joint1', im_out1)
    cv2.imshow('joint2', im_out2)

    img2 = ((im_out/255)+(im_out1/255)+(im_out2/255))/3
    mask2 = np.where(img2 == im_out2,img2,im_out2)
    mask1 = np.where(img2 == im_out1,img2,im_out1)
    im_out1 = mask1*(im_out1/255)/2
    im_out2 = mask2*(im_out2/255)/2
    img2 = ((im_out/255)+(im_out1/255)+(im_out2/255))

    cv2.imshow('joint3', img2)
    cv2.imwrite('joint.jpg', img2)
    img2 = cv2.imread('joint.jpg')

    # cv2.imshow('joint', im_out)
    # cv2.imshow('joint1', im_out1)
    # cv2.imshow('joint2', im_out2)

        
    


    video_out.write(image0)
    video_out1.write(image1)
    video_out2.write(image2)

    
    cv2.waitKey(1)            
        # release resources
cv2.destroyAllWindows()
# video_out.release()
