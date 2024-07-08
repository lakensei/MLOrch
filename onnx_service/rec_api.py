import base64
import copy
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse

from onnx_service.rapidocr_onnxruntime.ch_ppocr_v3_det import TextDetector
from onnx_service.rapidocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
from onnx_service.rapidocr_onnxruntime.utils import read_yaml

app = FastAPI()

ROOT_PATH = Path(__file__).resolve().parent

DET_CONFIG_PATH = ROOT_PATH.joinpath("rapidocr_onnxruntime").joinpath("ch_ppocr_v3_det")
det_config: Dict[str, Any] = read_yaml(DET_CONFIG_PATH / "config.yaml")

DET_MODEL_PATH = ROOT_PATH / "model" / "det_model.onnx"
det_config["model_path"] = DET_MODEL_PATH
det_session = TextDetector(det_config)

REC_CONFIG_PATH = ROOT_PATH.joinpath("rapidocr_onnxruntime").joinpath("ch_ppocr_v3_rec")
rec_config: Dict[str, Any] = read_yaml(REC_CONFIG_PATH / "config.yaml")

REC_MODEL_PATH = ROOT_PATH / "model" / "rec_model.onnx"
REC_KEYS_PATH = ROOT_PATH / "model" / "ppocr_keys_v1.txt"
rec_config["model_path"] = REC_MODEL_PATH
rec_config["rec_keys_path"] = REC_KEYS_PATH
rec_session = TextRecognizer(rec_config)


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


def sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                    abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                    and _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.array(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    ).astype(np.float32)
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class ModelReq(BaseModel):
    image: str


@app.get("/")
async def root():
    return JSONResponse(content={"data": "PaddleOCR det+rec model"})


@app.post("/ocr")
async def ocr(data: ModelReq):
    img = preprocess_image(data.image)
    ori_im = img.copy()
    dt_boxes, predict_time = det_session(img)
    if dt_boxes is None or len(dt_boxes) < 1:
        return None, 0.0
    dt_boxes = sorted_boxes(dt_boxes)
    img_crop_list = []
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
    rec_res, elapse = rec_session(img_crop_list)
    res = [
        {
            "bbox": tuple((int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1]))),
            "text": res[0],
            "score": float(res[1])
        } for box, res in zip(dt_boxes, rec_res)
    ]
    return JSONResponse(
        content={"data": {"res": res, "det_time": predict_time, "rec_time": elapse, "time": predict_time + elapse}})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
