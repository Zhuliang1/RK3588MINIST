import os
import cv2
import numpy as np
from rknn.api import RKNN

# Set your MNIST dataset path
MNIST_DATASET_PATH = './dataset.txt'

# Create RKNN object
rknn = RKNN()

# pre-process config
print('--> Config model')

rknn.config(
#0.13065974414348602 0.3015038073062897
    target_platform="rk3588s",mean_values=[[0.13065974414348602*255]], std_values=[[0.3015038073062897*255]]
)  # Quantize input data

print('done')

# Load ONNX model (MNIST model)
print('--> Loading model')
ret = rknn.load_onnx(model='mnist_model.onnx')  # Load your pre-trained MNIST model
if ret != 0:
    print('Load MNIST model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False, dataset=MNIST_DATASET_PATH)  # Build with MNIST dataset
if ret != 0:
    print('Build MNIST model failed!')
    exit(ret)
print('done')

# Export RKNN model
print('--> Export RKNN model')
ret = rknn.export_rknn('mnist_model.rknn')
if ret != 0:
    print('Export MNIST model to RKNN format failed!')
    exit(ret)
print('done')

# Initialize runtime environment
print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')
example_image = cv2.imread('mnist_image6.jpg', cv2.IMREAD_GRAYSCALE)
#
# example_image = np.expand_dims(example_image, axis=-1)  # Add channel dimension
# img=example_image.astype(np.float32)
# img = np.expand_dims(img, 0).astype(np.float32)
# Convert BGR to RGB
#rgb_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Convert RGB to grayscale
#gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
# cv2.imshow('Gray Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load the example image in grayscale mode
# example_image = cv2.imread('mnist_image6.jpg', cv2.IMREAD_GRAYSCALE)
# # Convert to HWC tensor
# example_image = np.expand_dims(example_image, axis=-1)  # Add channel dimension
# #example_image = example_image / 255.0  # Normalize to [0, 1] range
# example_image=example_image.astype(np.float32)
# Resize the image to match the model's input size
# input_height, input_width = 28, 28
# resized_image = cv2.resize(example_image, (input_width, input_height))

# Normalize the image data
#normalized_image = resized_image / 255.0

# Reshape the image to match the expected input shape (1, 28, 28, 1)
# Convert the image data to float32
# input_data = np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=3).astype(np.float32)
# img = np.expand_dims(resized_image, 2).astype(np.float32)


#img = np.expand_dims(img, 0).astype(np.float32)
# print(gray_image.shape)
# gray_image = np.expand_dims(gray_image, axis=-1)  # Add channel dimension
# Inference
print('--> Running model')
outputs = rknn.inference(inputs=[example_image],data_format="nchw")[0]
print(outputs)
# Apply softmax to the output
#softmax_outputs = np.argmax(outputs[0]
classes = np.argmax(outputs, axis=-1)

print("Predicted class:", classes)

# Release the RKNN object
rknn.release()

