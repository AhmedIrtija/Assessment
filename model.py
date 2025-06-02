from pytorch_model import Classifier, BasicBlock
from PIL import Image
import onnxruntime as ort
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.model = Classifier(BasicBlock, [2, 2, 2, 2])

    #From the pytorch_model file aand just converting it to numpy
    def preprocess(self, imgPath):
        img = Image.open(imgPath).convert("RGB")
        inp = self.model.preprocess_numpy(img).unsqueeze(0).numpy()
        return inp
    
# Taken straight from OnnxRuntime documentation
class ONNXModel:
    def __init__(self, model_path="model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.inputName = self.session.get_inputs()[0].name

    # Check for numpy then just get output which is also from the doc
    def predict(self, input):
        if not isinstance(input, np.ndarray):
            raise TypeError("Expected input to be a NumPy array")

        outputs = self.session.run(None, {self.inputName: input})
        return np.argmax(outputs[0])
