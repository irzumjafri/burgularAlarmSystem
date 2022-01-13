import cv2
import argparse
import numpy as np
import imageio
import tqdm
import subprocess

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mode', required=True,
                help = 'offline or online')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



if args.mode == "offline":
    video_out = cv2.VideoWriter('offline_output.avi',cv2.VideoWriter_fourcc('M','P','4','2'), 10, (640,480))
    results= open("results_offline.txt","w+")
    print("Enter Video File Name: ")
    x = input()
    video_file = x
    rate = 10
    vid_to_image = []
    reader = imageio.get_reader(video_file)
    meta_data = reader.get_meta_data()
    fps = meta_data['fps']
    n_frames = meta_data['nframes']
    for i, img in tqdm.tqdm(enumerate(reader), total=n_frames):
        if rate is None or i % int(round(fps / rate)) == 0:
            vid_to_image.append(img)

    for i in range (0, len(vid_to_image)):
        image = vid_to_image[i]
        Width = image.shape[1]
        Height = image.shape[0]
        classes = None
        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet(args.weights, args.config)
        blob = cv2.dnn.blobFromImage(image, 0.00392, (640,480), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold, nms_threshold = 0.5, 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y  = int(detection[0] * Width), int(detection[1] * Height)
                    w, h = int(detection[2] * Width), int(detection[3] * Height)
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
            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        cv2.imshow("Offline Video", image)
        video_out.write(image)
        cv2.waitKey(1)
            
    cv2.destroyAllWindows()

elif args.mode == "online":
    subprocess.call([r'yolo.bat'])
