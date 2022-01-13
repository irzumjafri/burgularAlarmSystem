@ECHO OFF 
:: This batch file details Windows 10, hardware, and networking configuration.
TITLE YOLO ONLINE
ECHO Switching on Camera 1
ECHO Switching on Camera 2
ECHO Switching on Camera 3
start python3 yolo_cam0.py --mode online --config cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights --classes data/coco.names
start python3 yolo_cam1.py --mode online --config cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights --classes data/coco.names
start python3 yolo_cam2.py --mode online --config cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights --classes data/coco.names
EXIT