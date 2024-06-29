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

# Function to preprocess the input image
def preprocess(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

# Function to postprocess the output
def postprocess(output):
    output = output.squeeze(0)
    output_predictions = output.argmax(0)
    return output_predictions.byte().cpu().numpy()

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, "tmp/fcn_resnet50.onnx", verbose=True, opset_version=11)

# Load the ONNX model in OpenCV
net = cv.dnn.readNetFromONNX('tmp/fcn_resnet50.onnx')

# Load and preprocess the image
image = cv.imread('data/2007_000033.jpg')
blob = cv.dnn.blobFromImage(image, scalefactor=1/255.0, size=(256, 256), mean=(0.485, 0.456, 0.406), swapRB=True, crop=False)
net.setInput(blob)

# Run forward pass
out = net.forward()

# Postprocess the output
out = out.transpose(0, 2, 3, 1)
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
print(f"seg {seg_width} {seg_height}")
print(f"img {img_width} {img_height}")
resized_img = cv.copyMakeBorder(src=image,
                                top=(seg_height - img_height)//2,
                                bottom=ceildiv((seg_height - img_height),2),
                                left=(seg_width - img_width)//2,
                                right=ceildiv((seg_width - img_width),2),
                                borderType=cv.BORDER_CONSTANT,
                                value=(0,0,0),
                               )
# alpha = 0.5
# blended = cv.addWeighted(image, alpha, seg_image, 1 - alpha, 0)

# Save and display the result
# cv.imwrite('segmentation_result.png', blended)
cv.imshow('Segmentation', seg_image)
cv.waitKey(0)
cv.destroyAllWindows()
