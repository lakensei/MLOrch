"""
python -m vllm.entrypoints.openai.api_server \
--model facebook/opt-125m \
--chat-template ./examples/template_chatml.jinja
"""
import argparse
import ssl
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

"""
response.replace("```", "").replace("json", "")
解析 ```json ```格式的数据

"""
import json
import re
from typing import List, Dict


class JsonParse:

    @staticmethod
    def _remove_comment(s):
        """移除JSON字符串中的错误注释（以//开头的行内注释）"""
        cleaned_str = re.sub(r'//.*?\n', '', s, flags=re.MULTILINE)
        return cleaned_str

    @staticmethod
    def _comma_correction(s):
        """存在中文逗号替换为英文, 多余的逗号"""
        return s.replace("\n", "").replace(" ", "").replace('"，', '",').replace(",}", "}")

    @staticmethod
    def load_markdown_json(markdown_text):
        """使用正则表达式匹配```json...```之间的内容"""
        pattern = r'```json(.*?)```'
        match = re.search(pattern, markdown_text, re.DOTALL)

        if match:
            # 获取匹配到的第一组内容（即```json...```之间的部分）
            json_str = match.group(1)
            json_str = JsonParse._remove_comment(json_str)
            json_str = JsonParse._comma_correction(json_str)
            try:
                # 使用json.loads将提取的字符串转换为Python对象
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"解析JSON时出错: {e}")
                return None
        else:
            print("未找到```json...```格式的字符串")
            return None


class IeModified:

    @staticmethod
    def remove_bracket(s):
        """
        从字符串中移除所有括号及其内部的内容。

        参数:
        s (str): 原始字符串。

        返回:
        str: 移除括号及其内容后的字符串。
        """
        # 使用正则表达式匹配圆括号及其内部的所有内容，包括嵌套括号
        # 注意：此正则表达式不处理嵌套括号的情况，对于简单的去括号需求足够
        s = s.replace("(", "（").replace(")", "）")
        pattern = r'\（[^()]*\）'

        # 使用递归替换，确保移除所有嵌套的括号对
        while re.search(pattern, s):
            s = re.sub(pattern, '', s)
        return s

    @staticmethod
    def extract_percentages(s):
        """
        提取字符串中的所有百分数。

        参数:
        text (str): 需要搜索的字符串。

        返回:
        list: 包含所有找到的百分数的列表。
        """
        # 正则表达式匹配百分数，例如 60%
        pattern = r'\d+(?:\.\d+)?%'
        # pattern = r'\d+\.?\d*%'
        # pattern = r'\b\d+(?:\.\d+)?\b'

        percentages = re.findall(pattern, s)
        # 将匹配到的百分比字符串转换为实际的浮点数比例（可选）
        # 如果需要转换，取消下面这行的注释
        # percentages = [float(p.strip('%')) / 100 for p in percentages]
        return percentages

    @staticmethod
    def check_ratio(s: List | str) -> List[str]:
        """
        存在格式：
            100% (全部结清)
            100%（结清）
            100% (支付剩余的合同价款，提供6%税率的税务发票)
            100% (支付剩余的合同价款，提供6%税率的税务发票)
            30%、30%、20%
            35%, 20%
            剩余款项剩余款项
            无息
        """
        if isinstance(s, List):
            s_list = []
            for s_ in s:
                s_ = IeModified.remove_bracket(s_)
                s_ = IeModified.extract_percentages(s_)
                if isinstance(s_, str):
                    s_list.append(s_)
                else:
                    # fixme: 如果仍为列表，说明解析失败   例如：["50%,40%", "30%"]
                    return []

            return s_list
        else:
            s = IeModified.remove_bracket(s)
            return IeModified.extract_percentages(s)

    @staticmethod
    def _split_str_node(s: str, n: int):
        """
        存在情况：
            一个payment_node中包含多个节点， 可能以\n ； 、,。等进行分割， 或直接为列表
            一个project_name中包含多个项目名， 可能以\n ; 、,。等进行分割， 或直接为列表
            project_name 获取失败的情况
            项目名称同时出现  智能物流管理系统开发项目/智能物流管理系统/本项目的情况
        """
        if isinstance(s, List):
            ...
        else:
            s1 = s.split('\n')
            if len(s1) == n:
                return s1
            s2 = s.split(';')
            if len(s2) == n:
                return s2
            s3 = s.split('、')
            if len(s3) == n:
                return s3

    @staticmethod
    def _cumulative_check(ratio_list) -> List | None:
        # 除第一个值外都为累计值
        transformed = [ratio_list[0]]
        for i in range(1, len(ratio_list)):
            diff = ratio_list[i] - ratio_list[i - 1]
            if diff <= 0:
                break
            transformed.append(diff)
        if sum(transformed) == 100:
            return transformed

    @staticmethod
    def readjust_ratio(ratio_list):
        total_ratio = sum(ratio_list)
        ratio_n = len(ratio_list)
        i_list = list(range(ratio_n))

        if total_ratio == 100:
            p = 100
            return i_list, ratio_list, p
        elif ratio_n <= 2 or total_ratio < 100:
            p = 30
            return i_list, ratio_list, p
        elif ratio_n > 3 and total_ratio == 200 and ratio_list[-1] == 100:
            p = 90
            return i_list[:-1], ratio_list[:-1], p
        elif ratio_n > 3 and sum(ratio_list[:-1]) == 100:
            p = 60
            return i_list[:-1], ratio_list[:-1], p
        else:

            if ratio_list[-1] == 100:
                # 最后一个值为累计值
                left_ratio = sum(ratio_list[:-1])
                if left_ratio < 100:
                    ratio_list[-1] = 100 - left_ratio
                    p = 60
                    return i_list, ratio_list, p
                # 从第二个值开始均为累计值
                transformed = IeModified._cumulative_check(ratio_list)
                if transformed:
                    p = 70
                    return i_list, transformed, p
            if ratio_list[-2] + ratio_list[-1] == 100:
                # 从第二个值开始均为累计值 但 最后一个非累计值
                ratio_list_ = [ratio_list[i] if i < ratio_n - 1 else 100 for i in range(ratio_n)]
                transformed = IeModified._cumulative_check(ratio_list_)
                if transformed:
                    p = 60
                    return i_list, transformed, p
            # 存在重复数据
            left_ratio_list = []  # 剔除相同数据后的左边比例
            left_ratio_i = []
            new_ratio_list = []  # 新结果
            new_ratio_i = []
            for i in i_list[:-1]:
                if len(left_ratio_list) == 0 or left_ratio_list[:-1] != ratio_list[i]:
                    left_ratio_list.append(ratio_list[i])
                    left_ratio_i.append(i)
                if ratio_list[i] == ratio_list[i + 1]:
                    if sum(left_ratio_list + ratio_list[i + 2:]) == 100:
                        new_ratio_list = left_ratio_list + ratio_list[i + 2:]
                        new_ratio_i = left_ratio_i + i_list[i + 2:]
                        break
                    if sum(ratio_list[:i + 1] + ratio_list[i + 2:]) == 100:
                        new_ratio_list = ratio_list[:i + 1] + ratio_list[i + 2:]
                        new_ratio_i = i_list[:i + 1] + i_list[i + 2:]
                        break
            if new_ratio_i:
                p = 80
                return new_ratio_i, new_ratio_list, p
            p = 10
            return i_list, ratio_list, p

    @staticmethod
    def handler(response):
        json_response = JsonParse.load_markdown_json(response)
        print(json_response)
        new_response: List[Dict] = []
        res = {"result": new_response, "p": 0}
        if json_response is None:
            return res
        else:
            if isinstance(json_response, dict):
                json_response = [json_response]
            for item in json_response:
                ratio_list = []
                node_list = []
                if not item.get("payments"):
                    continue
                for payment in item["payments"]:
                    if not isinstance(payment.get("payment_ratio"), str) or not isinstance(payment.get("payment_node"),
                                                                                           str):
                        continue
                    payment_ratio = IeModified.check_ratio(payment["payment_ratio"])
                    if len(payment_ratio) != 1:
                        continue
                    now_ratio = float(payment_ratio[0][:-1])
                    ratio_list.append(now_ratio)
                    node_list.append(payment["payment_node"])
                i_list, new_ratio_list, p = IeModified.readjust_ratio(ratio_list)
                payments = [{"payment_node": node_list[i], "payment_ratio": new_ratio_list[i]} for i in i_list]
                project_name = "本项目" if item.get("project_name", "项目名称") == "项目名称" else item["project_name"]
                new_response.append(
                    {"project_name": project_name, "payments": payments, "p": p}
                )
        res["result"] = new_response
        if new_response:
            res["p"] = sum([project["p"] for project in new_response]) / len(new_response)
        return res

    @staticmethod
    def handler1(response):
        is_modify = False
        p = 10
        # 对数据进行调整
        json_response = JsonParse.load_markdown_json(response)
        for item in json_response:
            project_info = {"project_name": item["project_name"]}
            payments = []
            ratio_list = []
            for payment in item["payments"]:
                payment_ratio: List[str] = IeModified.check_ratio(payment["payment_ratio"])
                if len(payment_ratio) != 1:
                    continue
                now_ratio = float(payment_ratio[0][:-1])
                ratio_list.append(now_ratio)
            total_ratio = sum(ratio_list)
            ratio_n = len(ratio_list)
            if total_ratio == 100:
                p = 100
            elif ratio_n < 2:
                p = 50
            elif total_ratio < 100:
                p = 50
            else:
                # 大于100， 分情况为累积和获取重复  (累积并且获取重复的情况无法处理)
                left_ratio_list = []  # 剔除相同数据后的左边比例
                new_ratio_list = []
                for i in range(ratio_n - 1):
                    if len(left_ratio_list) == 0 or left_ratio_list[:-1] != ratio_list[i]:
                        left_ratio_list.append(ratio_list[i])
                    if ratio_list[i] == ratio_list[i + 1]:
                        if sum(left_ratio_list + ratio_list[i + 2:]) == 100:
                            new_ratio_list = left_ratio_list + ratio_list[i + 2:]
                            break
                        if sum(ratio_list[:i + 1] + ratio_list[i + 2:]) == 100:
                            new_ratio_list = ratio_list[:i + 1] + ratio_list[i + 2:]
                            break
                if not new_ratio_list:
                    # 1. 判断是否累积
                    if ratio_list[-1] == 100:
                        ...
                        # [30, 40, 100]  [30, 70, 100]
                        for i in range(ratio_n, 0, -1):
                            ratio = ratio_list[i] - ratio_list[i + 1]
                            if ratio <= 0:
                                continue
                            if sum(ratio_list[i:]) + ratio == 100:
                                ratio_list[i] = ratio
                                break




                    elif sum(ratio_list[-2:]) == 100:
                        # [... , 95, 5]
                        ...
                    else:
                        for i in range(ratio_n - 1):
                            if ratio_list[i] < ratio_list[i + 1]:
                                ratio_list[i + 1] = ratio_list[i + 1] - ratio_list[i]


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*']
                   )


class ChatReq(BaseModel):
    max_length: int = Field(8192, title='生成文本的最大长度  0, 32768, 8192')
    top_p: float = Field(0.7, title='生成文本的多样性  0.0, 1.0, 0.8')
    temperature: float = Field(0.6, title='生成文本的随机性 0.0, 1.0, 0.6')
    role: str = Field('user', title='输入文本角色，system/user/assistant/observation')
    query: str = Field(..., title='用户输入的文本')
    history: List[dict] = Field([], title='对话历史')
    csv_file_id: str = Field(None, title='csv文件id')
    stream: Optional[bool] = False


class VllmModelReq(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: Optional[float] = None
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    # response_format: Optional[ResponseFormat] = None
    # seed: Optional[int] = Field(None,
    #                             ge=torch.iinfo(torch.long).min,
    #                             le=torch.iinfo(torch.long).max)
    # stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 1.0
    user: Optional[str] = None


model_dict = {
    "Qwen-14B-Chat-Int8": {
        "model_path": "/root/autodl-tmp/Qwen/Qwen-14B-Chat-Int8",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1"
    },
    "Qwen15-7B-Chat": {
        "model_path": "/root/autodl-tmp/Qwen/Qwen1___5-7B-Chat",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1"
    },
    "Qwen15-14B-Chat-AWQ": {
        "model_path": "/root/autodl-tmp/Qwen/Qwen1___5-14B-Chat-AWQ",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1"
    },
    "Qwen15-14B-Chat-GPTQ-Int4": {
        "model_path": "/root/autodl-tmp/Qwen/Qwen1___5-14B-Chat-GPTQ-Int4",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1"
    }
}

model: str
vllm_client: OpenAI


def init_model(model_name: str):
    model_info = model_dict[model_name]
    global vllm_client
    global model
    model = model_info['model_path']
    vllm_client = OpenAI(
        api_key=model_info['api_key'],
        base_url=model_info['base_url']
    )


init_model('Qwen15-14B-Chat-AWQ')


# @app.get("/health")
# async def health() -> Response:
#     """Health check."""
#     return Response(status_code=200)


@app.get("/ml/switch")
async def ml_switch(model_name: str = Query(..., title='模型名称')) -> Response:
    init_model(model_name)
    return Response(status_code=200)


@app.get("/ml/health")
async def ml_health() -> Response:
    return Response(content=200)


@app.post("/v1/chat/completions")
async def chat(query: VllmModelReq) -> Response:
    # messages = query.history + [{"role": query.role, "content": query.query}]
    chat_completion = vllm_client.chat.completions.create(
        messages=query.messages,
        model=model,
    )

    response = chat_completion.choices[0].message.content
    print(response)
    return JSONResponse({
        "code": 200,
        "data": {
            "origin_data": response,
            **IeModified.handler(response)
        }
    })


"""

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    args = parser.parse_args()
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
