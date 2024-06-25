import sys
import os
from pathlib import Path
sys.path.append(Path(os.path.dirname(os.path.realpath(__file__))).parent)

import solInfer as solInfer
from solInfer.models.resnet50 import ResNet50
import cv2
import numpy as np

def get_preprocessed_img(img_path: str) -> solInfer.backends.Tensor:
    # read the image
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)

    input_img = cv2.resize(input_img, (256, 256))

    # define preprocess parameters
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(224, 224),  # img target size
        mean=mean,
        swapRB=True,  # BGR -> RGB
        crop=True  # center crop
    )
    # 3. divide by std
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return solInfer.backends.NPTensor(input_blob)

if __name__ == "__main__":
    # Get the ResNet50 model and the imagenet labels
    torch_model = ResNet50.getPretrained()
    imagenet_labels = ResNet50.loadImagenetLabelsFromFile("data/classification_classes_ILSVRC2012.txt")

    # Convert the model to ONNX and load it with OpenCV DNN
    onnx_filename = ResNet50.convertToDiskFormat(solInfer.models.diskFormats.ONNX)
    cv_model = solInfer.backends.cvDNN.Net.loadONNX(onnx_filename)

    input_img = get_preprocessed_img("data/squirrel_cls.jpg")

    # Run with OpenCV DNN
    result = ResNet50.infer(cv_model, input_img, imagenet_labels)
    print(result)

    # Run with PyTorch DNN
    result = ResNet50.infer(torch_model, input_img, imagenet_labels)
    print(result)