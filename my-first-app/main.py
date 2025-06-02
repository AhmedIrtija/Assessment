from model import ImagePreprocessor, ONNXModel
from fastapi import FastAPI, UploadFile, File
import shutil

#Initialize app
app = FastAPI()

# set the objects
model = ONNXModel("model.onnx")
preprocessor = ImagePreprocessor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    tensor = preprocessor.preprocess(path)
    prediction = model.predict(tensor)
    return {"class_id": int(prediction)}
