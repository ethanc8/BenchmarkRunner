import torch
import torch.onnx
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

def ceildiv(n, d):
    return -(n // -d)

# Load a pre-trained PyTorch model (FCN ResNet50)
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
model.eval()

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 500, 500)
torch.onnx.export(model, dummy_input, "tmp/fcn_resnet50.onnx", verbose=True, opset_version=11)

# Load the ONNX model in OpenCV
net = cv.dnn.readNetFromONNX('tmp/fcn_resnet50.onnx')

# Load and preprocess the image
image = cv.imread('data/2007_000033.jpg')
blob = cv.dnn.blobFromImage(image, scalefactor=1/255.0, size=(500, 500), mean=np.array([0.485, 0.456, 0.406]) * 255.0, swapRB=True, crop=False)
blob[0] /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
net.setInput(blob)

# Run forward pass
out = net.forward()

# Postprocess the output
# (batch_size == 1, num_classes, height, width) -> (1, height, width, num_classes)
out = out.transpose(0, 2, 3, 1)
# (1, height, width, num_classes) -> (height, width)
seg_map = np.argmax(out, axis=3)[0]

# Map each label to a color
colors = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                   [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                   [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])

# Create segmentation image
seg_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
for label in range(len(colors)):
    seg_image[seg_map == label] = colors[label]

# # Blend the original image with the segmentation image
seg_width = seg_image.shape[1]
seg_height = seg_image.shape[0]
img_width = image.shape[1]
img_height = image.shape[0]
blob_width = blob.shape[1]
blob_height = blob.shape[0]
print(f" seg {seg_width} {seg_height}")
print(f" img {img_width} {img_height}")
print(f"blob {blob_width} {blob_height}")
# OpenCV pads the bottom and right more than the top and left if the top and bottom have
# different padding, when doing blobFromImage, so we'll do the same here.
resized_img = cv.copyMakeBorder(src=image,
                                top=(seg_height - img_height)//2,
                                bottom=ceildiv((seg_height - img_height),2),
                                left=(seg_width - img_width)//2,
                                right=ceildiv((seg_width - img_width),2),
                                borderType=cv.BORDER_CONSTANT,
                                value=(0,0,0),
                               )
alpha = 0.5
blended = cv.addWeighted(resized_img, alpha, seg_image, 1 - alpha, 0)

# Save and display the result
# cv.imwrite('segmentation_result.png', blended)
cv.imshow('Segmentation', blended)
cv.waitKey(0)
cv.destroyAllWindows()
