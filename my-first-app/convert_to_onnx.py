import torch
from pytorch_model import Classifier, BasicBlock

# initialize the weights using pytorch_model.py
model = Classifier(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
model.load_state_dict(torch.load("weights/pytorch_model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# Batch size, RGB, and image sizes
dummy_input = torch.randn(1, 3, 224, 224)

# Export the correct model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
