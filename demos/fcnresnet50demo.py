# SPDX-License-Identifier: AGPL-3.0-or-later
# A lot of this code is from https://docs.opencv.org/4.x/d7/d9a/pytorch_segm_tutorial_dnn_conversion.html
import sys
import os
import solInfer as solInfer
from solInfer.models.fcnresnet50 import FCNResNet50
import cv2
import numpy as np

def get_processed_imgs(img_path: str):
    # read the image
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = input_img
    input_img = input_img.astype(np.float32)
    # target image sizes
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    
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
        size=(img_width, img_height), # img target size
        mean=mean,
        swapRB=True, # BGR -> RGB
        crop=False # center crop
    )
    # 3. divide by std
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    return img, solInfer.backends.NPTensor(input_blob)

def read_colors_info(filename):
    pascal_voc_classes = []
    pascal_voc_colors = []
    with open(filename) as f:
        for line in f.readlines():
            name, r, g, b = line.split()
            pascal_voc_classes.append(name)
            pascal_voc_colors.append((r, g, b))
    return pascal_voc_classes, pascal_voc_colors

# def get_colored_mask(img_shape, segm_mask, pascal_voc_colors):
#     img_width = img_shape[1]
#     img_height = img_shape[0]
#     print(f"segm.mask shape: {segm_mask.shape}")
#     # convert mask values into PASCAL VOC colors
#     processed_mask = np.stack([pascal_voc_classes[color_id] for color_id in segm_mask.flatten()])
#     print(f"segm.mask shape: {segm_mask.shape}")
#     mask_width = segm_mask.shape[1]
#     mask_height = segm_mask.shape[0]

#     # reshape mask into 3-channel image
#     processed_mask = processed_mask.reshape(mask_height, mask_width, 3)
#     processed_mask = cv2.resize(processed_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST).astype(
#     np.uint8)
    
#     # convert colored mask from BGR to RGB for compatibility with PASCAL VOC colors
#     processed_mask = cv2.cvtColor(processed_mask, cv2.COLOR_BGR2RGB)
#     return processed_mask

def get_colored_mask(img_shape, prediction, pascal_voc_colors):
    img_width = img_shape[1]
    img_height = img_shape[0]
    print(f"img shape: {img.shape}")
    print(f"prediction shape: {prediction.shape}")

    predicted_classes = prediction.argmax(axis=0)
    print(f"predicted_classes shape: {predicted_classes.shape}")

    # Initialize an empty array for RGB image
    h, w = predicted_classes.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for x in range(img_width):
        for y in range(img_height):
            colored_mask[y,x] = pascal_voc_colors[predicted_classes[y,x]]

    # # Assign colors based on predicted_classes and pascal_voc_colors
    # for class_index in range(len(pascal_voc_colors)):
    #     # Get all pixels where predicted_classes == class_index
    #     mask = predicted_classes == class_index
    #     # Assign the corresponding color to all these pixels
    #     colored_mask[mask] = pascal_voc_colors[class_index]

    # note: these should probably work the same, but I'm just keeping both options just in case.

    return colored_mask


if __name__ == "__main__":
    # Get the ResNet50 model and the imagenet labels
    torch_model = FCNResNet50.get_pretrained()

    # Convert the model to ONNX and load it with OpenCV DNN
    os.makedirs("tmp/", exist_ok=True)
    onnx_filename = FCNResNet50.convert_to_disk_format(solInfer.models.disk_formats.ONNX, torch_model, "tmp/resnet50.onnx")
    cv_model = solInfer.backends.cvDNN.Net.loadONNX(onnx_filename)

    img, input_img = get_processed_imgs("data/2007_000033.jpg")
     
    pascal_voc_classes, pascal_voc_colors = read_colors_info("data/pascal-classes.txt")

    # Run with OpenCV DNN
    print("Inferring with OpenCV DNN...")
    opencv_prediction = FCNResNet50.infer(cv_model, input_img)

    # Run with PyTorch
    print("Inferring with PyTorch...")
    pytorch_prediction = FCNResNet50.infer(torch_model, input_img)

    # obtain colored segmentation masks
    print(f"img.shape: {img.shape}")
    opencv_colored_mask = get_colored_mask(img.shape, opencv_prediction, pascal_voc_colors)
    pytorch_colored_mask = get_colored_mask(img.shape, pytorch_prediction, pascal_voc_colors)
    
    # obtain palette of PASCAL VOC colors
    # color_legend = get_legend(pascal_voc_classes, pascal_voc_colors)
    
    cv2.imshow('PyTorch Colored Mask', pytorch_colored_mask)
    cv2.imshow('OpenCV DNN Colored Mask', opencv_colored_mask)
    # cv2.imshow('Color Legend', color_legend)
    
    cv2.waitKey(0)