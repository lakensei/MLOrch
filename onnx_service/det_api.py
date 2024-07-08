import base64
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse

from onnx_service.rapidocr_onnxruntime.ch_ppocr_v3_det.text_detect import TextDetector
from onnx_service.rapidocr_onnxruntime.utils import read_yaml

app = FastAPI()

ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_PATH.joinpath("rapidocr_onnxruntime").joinpath("ch_ppocr_v3_det")
config: Dict[str, Any] = read_yaml(CONFIG_PATH / "config.yaml")

MODEL_PATH = ROOT_PATH / "model" / "det_model.onnx"
config["model_path"] = MODEL_PATH
ort_session = TextDetector(config)


def preprocess_image(image_base64: str) -> np.ndarray:
    # 移除数据前缀
    image_data = image_base64.split(",")[1]
    # 解码Base64数据
    image_bytes = base64.b64decode(image_data)
    # 将字节数据转换为NumPy数组
    nparr = np.frombuffer(image_bytes, np.uint8)
    # 使用cv2将NumPy数组解码为图像
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


class ModelReq(BaseModel):
    image: str


@app.get("/")
async def root():
    return JSONResponse(content={"data": "PaddleOCR det model"})


@app.post("/ocr")
async def ocr(data: ModelReq):
    img = preprocess_image(data.image)
    det_res, predict_time = ort_session(img)
    res = [{"bbox": tuple((int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])))} for box in det_res]
    return JSONResponse(content={"data": res})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
