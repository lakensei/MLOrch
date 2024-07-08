# PaddleOCR 的onnx格式部署
包括版面分析、文本检测、文字识别

# paddle模型转为onnx  
> https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddle2onnx/readme.md

模型地址在`https://github.com/PaddlePaddle/PaddleOCR` 中进行获取

```shell
pip install paddle2onnx
pip install onnxruntime==1.9.0

# wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
# tar -vxf ch_PP-OCRv4_det_infer.tar
paddle2onnx --model_dir ./ch_PP-OCRv4_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./det_model.onnx \
--opset_version 10 \
--enable_onnx_checker True

# https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
# https://github.com/PaddlePaddle/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt
paddle2onnx --model_dir ./ch_PP-OCRv4_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./rec_model.onnx \
--opset_version 10 \
--enable_onnx_checker True

paddle2onnx --model_dir ./ch_ppstructure_mobile_v2.0_SLANet_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./table_model.onnx \
--opset_version 10 \
--enable_onnx_checker True

# wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar
paddle2onnx --model_dir ./picodet_lcnet_x1_0_fgd_layout_cdla_infer \
--model_filename model.pdmodel \
--params_filename model.pdiparams \
--save_file ./picodet_layout.onnx \
--opset_version 10 \
--enable_onnx_checker True
```

# onnx模型调用
> 代码来自 `https://github.com/RapidAI/RapidOCR/tree/main/python/rapidocr_onnxruntime`
> 这里只有cls、det、rec三个模型的调用， layout的代码是从一个issue中看到的

# 部署
```shell
# 版面分析
uvicorn onnx_service.layout_api:app --host 0.0.0.0 --port 10011 --workers 1

```