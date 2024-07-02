import argparse
import time
from pathlib import Path
from typing import Dict, Any, Union

import cv2
import numpy as np

from .utils import PicoDetPostProcess, PicoDetPreProcess
from ..utils import OrtInferSession, LoadImage, read_yaml


class StructureLayout:

    def __init__(self, config: Dict[str, Any]):
        self.session = OrtInferSession(config)
        self.preprocess_op = PicoDetPreProcess(config["pre_process"])
        self.postprocess_op = PicoDetPostProcess(config["label_list"], **config["post_process"])
        self.load_img = LoadImage()

    def __call__(self, img_content: Union[str, np.ndarray, bytes, Path]):
        img = self.load_img(img_content)

        ori_im = img.copy()
        data = self.preprocess_op(img)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        preds, elapse = 0, 1
        starttime = time.time()

        preds = self.session(img)

        score_list, boxes_list = [], []
        num_outs = int(len(preds) / 2)
        for out_idx in range(num_outs):
            score_list.append(preds[out_idx])
            boxes_list.append(preds[out_idx + num_outs])
        preds = dict(boxes=score_list, boxes_num=boxes_list)
        post_preds = self.postprocess_op(ori_im, img, preds)
        elapse = time.time() - starttime
        return post_preds, elapse
