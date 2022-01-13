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

video_out = cv2.VideoWriter("Alertvideo.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
video_out1 = cv2.VideoWriter("Alertvideo1.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
video_out2 = cv2.VideoWriter("Alertvideo2.avi",cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))


imgblack = np.zeros((512,512,3), np.uint8)
cam0_fix = np.array([[486, 3], [446, 2], [413, 261], [435, 326]])
cam0_fix = cam0_fix.reshape((-1,1,2))
cam1_fix = np.array([[350, 151], [280, 160], [288, 372], [355, 400]])
cam1_fix = cam1_fix.reshape((-1,1,2))
cam2_fix = np.array([[308, 0], [0, 76], [52, 357], [319, 283]])
cam2_fix = cam2_fix.reshape((-1,1,2))

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

    cv2.polylines(image0,cam0_fix,True,(255,255,255),2)
    cv2.polylines(image1,cam1_fix,True,(255,255,255),2)
    cv2.polylines(image2,cam2_fix,True,(255,255,255),2)

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
        draw_bounding_box(image0, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        if cv2.pointPolygonTest(cam0_fix, (x, y), False) > 0 or cv2.pointPolygonTest(cam0_fix, (x+w, y), False) > 0 or cv2.pointPolygonTest(cam0_fix, (x, y+h), False) > 0 or cv2.pointPolygonTest(cam0_fix, (x+w, y+h), False) > 0:
            cv2.putText(image0, "ALERT!!!", (round(x),round(y)+round(h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            video_out.write(image0)



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
        draw_bounding_box(image1, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        if cv2.pointPolygonTest(cam1_fix, (x, y), False) > 0 or cv2.pointPolygonTest(cam1_fix, (x+w, y), False) > 0 or cv2.pointPolygonTest(cam1_fix, (x, y+h), False) > 0 or cv2.pointPolygonTest(cam1_fix, (x+w, y+h), False) > 0:
            cv2.putText(image1, "ALERT!!!", (round(x),round(y)+round(h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            video_out1.write(image1)



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
        draw_bounding_box(image2, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        if cv2.pointPolygonTest(cam2_fix, (x, y), False) > 0 or cv2.pointPolygonTest(cam2_fix, (x+w, y), False) > 0 or cv2.pointPolygonTest(cam2_fix, (x, y+h), False) > 0 or cv2.pointPolygonTest(cam2_fix, (x+w, y+h), False) > 0:
            cv2.putText(image2, "ALERT!!!", (round(x),round(y)+round(h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            video_out2.write(image2)


    cv2.imshow("Live Cam 1", image0)
    cv2.imshow("Live Cam 2", image1)
    cv2.imshow("Live Cam 3", image2)

    # video_out.write(image0)
    # video_out1.write(image1)
    # video_out2.write(image2)

    
    cv2.waitKey(1)            
        # release resources
cv2.destroyAllWindows()
# video_out.release()