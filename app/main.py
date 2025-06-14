from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ocr_pipeline import process_image

app = FastAPI()
@app.post("/ocr/")
def ocr_api(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = process_image(image)
    if not result:
        return {
            "status": "fail",
            "message": "Không phát hiện được CMQS hoặc ảnh quá mờ"
        }
    return {
        "status": "success",
        "data": result
    }
