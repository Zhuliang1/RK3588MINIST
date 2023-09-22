import cv2
import random
import numpy as np
import timeit
import struct
from rknn.api import RKNN


# ... (定义相关的函数和参数)
def data_fetch_preprocessing():
    train_image = open('train-images.idx3-ubyte', 'rb')
    test_image = open('t10k-images.idx3-ubyte', 'rb')
    train_label = open('train-labels.idx1-ubyte', 'rb')
    test_label = open('t10k-labels.idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',
                             train_label.read(8))
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,
                                         dtype=np.uint8), ndmin=1)
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',
                                 test_label.read(8))
    y_test = np.fromfile(test_label,
                         dtype=np.uint8).reshape(10000, 1)
    # print(y_train[0])
    # 训练数据共有60000个
    # print(len(labels))
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784)

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784)
    # print(x_train.shape)
    # 可以通过这个函数观察图像
    # data=x_train[:,0].reshape(28,28)
    # plt.imshow(data,cmap='Greys',interpolation=None)
    # plt.show()

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train_label, x_test, y_test
# 定义前处理函数
def preprocess(input_data):
    blob = cv2.dnn.blobFromImage(input_data, scalefactor=1.0, size=(28, 28), mean=0.5, swapRB=False)
    return blob
ONNX_MODEL="mnist_model.onnx" 
DATASET="dataset.txt"   
RKNN_MODEL="mnist_model.rknn"  
    
if __name__ == '__main__':

    # 创建 RKNN 对象
    rknn = RKNN(verbose=True)
        # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config()
    
    print('done')
    
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    # 获取数据集长度
    dataset_length = len(x_test)

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
    for i in range(dataset_length):
    #for i in range(10):
        input_data = x_test[i].astype(np.float32).reshape(28, 28,1)
        #single_channel_image = np.expand_dims(input_image, axis=-1)
        input_image = input_data.astype(np.float32)
        #print(input_image)

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[input_image])
        softmax_outputs = np.argmax(outputs)
        print("Predicted class:", softmax_outputs)
        true_label = y_test[i][0]
        print("true_label: ",true_label)
        if softmax_outputs == true_label:
            correct_predictions += 1
        print(f"Image {i + 1} - Correct Prediction: {softmax_outputs == true_label}")

        # 处理推理结果并计算正确率
        # ...（根据模型输出进行处理和计算）

    # 计算正确率
    accuracy =accuracy = correct_predictions / dataset_length * 100
    print(f"Model Accuracy: {accuracy:.2f}%")

    rknn.release()

