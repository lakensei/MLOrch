# qwen2模型
采用vllm部署
> qwen_openai_api.py 为qwen模型的openai格式服务

# 模型下载
```shell
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct-GPTQ-Int4', cache_dir='/root/autodl-tmp')
```

# 部署
```shell
# qwen2大模型  vlm 支持touch==2.1.2需3.0.0版本
# python -m vllm.entrypoints.openai.api_server  --model /root/autodl-tmp/qwen/Qwen2-7B-Instruct-GPTQ-Int4 --served-model-name qwen2  --quantization gptq --port 8001
sh ./qwen2_service/run.sh
```