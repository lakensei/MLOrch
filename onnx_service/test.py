import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

from onnx_service.rapidocr_onnxruntime.ch_structure_v2_layout.utils import vis_layout


def layout_test():
    from onnx_service.rapidocr_onnxruntime.ch_structure_v2_layout.structure_layout import StructureLayout
    from onnx_service.rapidocr_onnxruntime.ch_structure_v2_layout.utils import vis_layout
    config = {
        'model_path': './model/picodet_layout.onnx',
        'use_cuda': False,
        'CUDAExecutionProvider': {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True
        },
        'pre_process': {
            'Resize': {'size': [800, 608]},
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.', 'order': 'hwc'
            },
            'ToCHWImage': None,
            'KeepKeys': {'keep_keys': ['image']}
        },
        'post_process': {'score_threshold': 0.5, 'nms_threshold': 0.5},
        'label_list': ["text", "title", "figure", "figure_caption", "table", "table_caption", "header", "footer",
                       "reference", "equation"]
    }
    text_recognizer = StructureLayout(config)

    img = cv2.imread(r"./1.jpeg")
    rec_res, predict_time = text_recognizer(img)
    vis_layout(img, rec_res, f'./1.png')
    print(f"rec result: {rec_res}\t cost: {predict_time}s")


def layout_api_test():
    import requests
    img = cv2.imread(r"./1.jpeg")
    _, buffer = cv2.imencode('.jpg', img)
    image_bytes = buffer.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')


    def send_request():
        return requests.post("http://127.0.0.1:8000/ocr", json={
            "image": f"data:image/jpeg;base64,{image_base64}"
        }).json()

    # 定义并发请求的数量
    num_requests = 100
    s = time.time()
    time_ = 0
    # 使用线程池进行并发请求
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]

        for future in as_completed(futures):
            try:
                response = future.result()
                time_ += response["data"]["time"]
            except Exception as e:
                print(f"Request generated an exception: {e}")
    e = time.time()
    print("time",e-s)
    print(">>>",time_)

def rec_test():
    config = {
        'model_path': './model/rec.onnx',
        'use_cuda': False,
        'CUDAExecutionProvider': {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True
        },
        "rec_img_shape": [3, 48, 320],
        "rec_batch_num": 6,
    }
    from onnx_service.rapidocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
    text_recognizer = TextRecognizer(config)
    img = cv2.imread(r"./1.jpeg")
    rec_res, predict_time = text_recognizer(img)
    vis_layout(img, rec_res, f'./1.png')
    print(f"rec result: {rec_res}\t cost: {predict_time}s")



if __name__ == '__main__':
    layout_api_test()
