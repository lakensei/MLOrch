import base64
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse

from onnx_service.rapidocr_onnxruntime.ch_structure_v2_layout.structure_layout import StructureLayout
from onnx_service.rapidocr_onnxruntime.utils import read_yaml

app = FastAPI()

ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_PATH.joinpath("rapidocr_onnxruntime").joinpath("ch_ppocr_v3_rec")
config: Dict[str, Any] = read_yaml(CONFIG_PATH / "config.yaml")

MODEL_PATH = ROOT_PATH / "model" / "rec.onnx"
config["model_path"] = MODEL_PATH
ort_session = StructureLayout(config)


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
    return JSONResponse(content={"data": "PaddleOCR layout model"})


@app.post("/ocr")
async def ocr(data: ModelReq):
    # img = np.array(data.image)
    # img_b64decode = base64.b64decode(data['image'])
    img = preprocess_image(data.image)
    layout_res, predict_time = ort_session(img)
    print(layout_res)
    return JSONResponse(content={"data": layout_res})


# @app.post("/ocr")
# async def ocr(image_file: UploadFile = None, image_data: str = Form(None)):
#     if image_file:
#         img = Image.open(image_file.file)
#     elif image_data:
#         img_bytes = str.encode(image_data)
#         img_b64decode = base64.b64decode(img_bytes)
#         img = Image.open(io.BytesIO(img_b64decode))
#     else:
#         raise ValueError(
#             "When sending a post request, data or files must have a value."
#         )
#
#     ocr_res = processor(img)
#     return ocr_res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
