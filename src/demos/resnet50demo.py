import src as solInfer
from src.models.resnet50 import ResNet50

if __name__ == "__main__":
    # Get the ResNet50 model and the imagenet labels
    torch_model = ResNet50.getPretrained()
    imagenet_labels = ResNet50.loadImagenetLabelsFromFile("data/")

    # Convert the model to ONNX and load it with OpenCV DNN
    onnx_filename = ResNet50.convertToDiskFormat(solInfer.models.diskFormats.ONNX)
    cv_model = solInfer.backends.cvDNN.Net.loadONNX(onnx_filename)

    # Run with OpenCV DNN
    result = ResNet50.infer(cv_model, imagenet_labels)
    print(result)

    # Run with PyTorch DNN
    result = ResNet50.infer(torch_model, imagenet_labels)
    print(result)