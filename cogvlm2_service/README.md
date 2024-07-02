# CogVLM2模型
OpenAI 格式部署服务
> 代码来源
> https://github.com/THUDM/CogVLM2/blob/main/basic_demo/openai_api_demo.py

# 模型下载
```shell
# https://www.modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B/files

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/cogvlm2-llama3-chinese-chat-19B', cache_dir='/root/autodl-tmp')
```
# 部署
```shell
# 本地运行
python cogvlm2_service\main.py
```

