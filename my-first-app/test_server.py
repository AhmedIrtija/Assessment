import requests
import time
import os
import sys


API_URL = "https://api.cortex.cerebrium.ai/v4/p-b747e048/my-first-app/predict"   
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWI3NDdlMDQ4IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoxNzUxNDE0NDAwfQ.P9r5JkfceuiwtFLla_kTyAQTwKq54rXQS4iBGiXSH33ZFPaCJVnHMscbJlALvrRA-mNMiPbxbvt19SCVwVFCEJ5D3hz-Grlt1Wg5P2rWHOXfZX68tbjjq2cY2YoIyVYZS3EQAfG3pLBtzUG7AYjn_c0izLcxiwbXKF0ftsXcHFoCghUk4pLAg0XVBLM7C8lyR3w0PiL0eyz0pNGIQG-lVNfTTdz70TwP2aNo63kS4XAwQWXijU3aAzEjSNjHMK3C6OruaZqdN6oc9ULErE86brrmE2FyrtAxWyu1cj-JZfRZip7SydDfyel3vdbaGalCHsqsCWALlcwfSUsXPly4FA"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# Function to send a prediction request to the API
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    # Open the image file and send it to the API
    with open(image_path, "rb") as f:
        files = {"file": f}
        start_time = time.time()
        try:
            response = requests.post(f"{API_URL}/", files=files, headers=HEADERS)
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return

        latency = time.time() - start_time

    if response.status_code != 200:
        print(f"[ERROR] Failed with status code {response.status_code}")
        print("Response:", response.text)
        return

    try:
        result = response.json()
        print(f"[PREDICTION] Class ID: {result}")
        print(f"[INFO] Inference Latency: {latency:.3f} sec")
    except ValueError:
        print("[ERROR] Failed to parse JSON response")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        predict(sys.argv[1])
    else:
        print("Usage:")
        print("  python test_server.py <path_to_image>")
        print("  python test_server.py --run-tests")
