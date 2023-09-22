import cv2
import random
import numpy as np
import timeit
import struct
from rknn.api import RKNN

ONNX_MODEL="test.onnx" 
DATASET="dataset.txt"   
RKNN_MODEL="test.rknn"  
    
if __name__ == '__main__':

    # 创建 RKNN 对象
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    #rknn.config(mean_values=[0], std_values=[255])
    rknn.config(target_platform='RK3588S')
    print('done')
    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 循环抽取的图像进行推理和计算正确率
    correct_predictions = 0
    #for i in range(dataset_length):
    for i in range(10):
        input_data = np.array([[i]], dtype=np.float32)  # 例如，假设模型期望的是一个 float32 类型的单个数值
        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[input_data])
        #softmax_outputs = np.argmax(outputs)
        print("True label:", i,"Predicted class:", outputs)
       

    rknn.release()

