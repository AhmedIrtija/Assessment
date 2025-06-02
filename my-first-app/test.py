from model import ImagePreprocessor, ONNXModel
import os

def test_model_inference(img_Path: str):
    try:
        preprocessor = ImagePreprocessor()
        model = ONNXModel("model.onnx")

    except Exception as e:
        print(f"[ERROR] Failed to initialize model or preprocessor: {e}")
        return

    if not os.path.exists(img_Path):
        print(f"[ERROR] Image not found at: {img_Path}")
        return

    try:
        img = preprocessor.preprocess(img_Path)

    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return

    try:
        pred_class = model.predict(img)
        print(f"Predicted class: {pred_class}")
        
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")


if __name__ == "__main__":
    test_img_Path = "assets/n01667114_mud_turtle.JPEG"
    test_model_inference(test_img_Path)