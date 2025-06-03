## Prerequisites

- Python 3.8 or above installed  
- `pip` for installing Python packages     

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Deliverables and Usage

### A. Model Preprocessing and Inference Code

- `pytorch_model.py`: Contains the PyTorch classifier definition and image preprocessing methods.  
- `onnx_model.py`: Loads the ONNX model and runs inference.  

These modules are used internally and for local testing.

---

### B. `test.py` — Local Model Testing Script

Use this to test your model locally before deployment.

```bash
python test.py <path_to_image>
```
It prints the predicted class ID for the given image.


### C. `test_server.py` — Testing the Deployed Model

This script tests your deployed model on Cerebrium by sending images and checking predictions.


#### Usage

To predict an image class ID:

```bash
python test_server.py <path_to_image>
```

To run platform monitoring tests:
```bash
python test_server.py --run-tests
```


## Contact 
ahmedirtija8@gmail.com