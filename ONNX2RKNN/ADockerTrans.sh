docker run -t -i --privileged -v /dev/bus/usb:/dev/bus/usb \
-v /home/zhuliang/Downloads/NEWRKNN/RKNN_Docker/rknn-toolkit-v1.7.3/examples/onnx/yolov5/minist:/rknn_yolov5_demo \
rknn-toolkit:1.7.3 /bin/bash
