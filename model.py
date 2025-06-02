from pytorch_model import Classifier, BasicBlock
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        self.model = Classifier(BasicBlock, [2, 2, 2, 2])

    #From the pytorch_model file aand just converting it to numpy
    def preprocess(self, imgPath):
        img = Image.open(imgPath).convert("RGB")
        inp = self.model.preprocess_numpy(img).unsqueeze(0).numpy()
        return inp