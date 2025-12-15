from typing import Dict

# FastAPI是一个类，可以用它来创建一个应用对象app
# app相当于一个“接线板/总控台“：
#   它维护了一张表：（HTTP方法，路径） -> 处理函数
#   请求来了就查表，找到处理函数并执行
from fastapi import FastAPI

# HTTPException 是一个异常类，专门用来“中断请求并返回指定的 HTTP 错误“
# 它的意义是告诉FastAPI别继续执行了，把“错误“用标准的HTTP方式返回给前端/调用方，而非让程序崩掉
from fastapi import HTTPException

# BaseModel来自 Pydantic，它是一个数据模型基类，用来定义“你希望收到/返回的数据长什么样“
from pydantic import BaseModel

# FastAPI是做什么的？
# 一句话：FastAPI 帮我把 Python 函数变成可通过 HTTP 调用的服务接口（API），并且自动做数据校验、文档生成
# 担任整个项目中间层的角色：
# 前端 -> HTTP 请求 -> FastAPI（中间层） -> 调用模型推理 -> HTTP 响应 -> 前端显示
# 前后端之间通过一个非常朴素的协议连接：HTTP + JSON

# 项目完整的链路图如下：
# (前端) 浏览器/Streamlit/HTML
#         |
#         |  HTTP 请求 (JSON)
#         v
# (后端) FastAPI 服务器  ----->  调用 inference.predict()  ----->  模型推理
#         |
#         |  HTTP 响应 (JSON)
#         v
# (前端) 展示 label/score

from inference import (
    InferenceConfig,
    load_model_and_tokenizer,
    predict as predict_fn,
)

# ===用Pydantic定义“接口的输入格式“和接口的输出格式“===
# 其实就分别像函数参数的类型签名和函数的返回值类型，只不过这里的函数调用是通过网络（HTTP）完成的，用 JSON 来承载数据

# TextInput定义了：接口期望收到一个 JSON，里面必须有一个字段text 字符串
# 作用A：自动解析 JSON -> Python 对象
# 作用B：自动校验+自动返回错误，检验请求体的合法性
class TextInput(BaseModel):
    text: str

# 作用A：保证返回的数据结构稳定
# 作用B：FastAPI会用它自动生成/docs文档，不用手写 API 文档，会自动同步更新
# 作用C：如果返回的数据非法，FastAPI能帮助尽快发现问题
class PredictOutput(BaseModel):
    label: str
    score: float
    probs: Dict[str, float]

# ==== 全局对象：FastAPI app + 模型相关 ====
# 创建一个“后端应用实例/总控台“。它的职责不是做“情感分类“，而是负责把情感分类能力包装成一个可被外界访问的 Web 服务
# app是一个HTTP请求的入口 + 路由分发器 + 生命周期管理器（以及一堆省事的自动化功能）
# app具体负责什么？
# A.监听并接收请求（准确说：作为被服务器调用的应用）
#   启动的是 uvicorn app.main:app --reload
#   uvicorn才是“真正监听端口的服务器“：接电话（监听端口、收 HTTP 请求）
#   app是被uvicorn调用的“应用对象“（ASGI应用）：接到电话后，决定谁来处理、怎么处理、怎么回话
# B.路由：把URL映射到你的函数
# C.数据解析与校验（借助 Pydantic），无需手动写 JSON 解析或校验代码
# D.响应构建：将函数输出统一变成 HTTP Response（默认 JSON）
# E.生命周期管理：启动时做初始化
# F.自动生成API文档
# app的职责：把模型能力“服务化“——让外界通过HTTP用到它。本质上是个“路由注册中心 + 请求分发器“

# 以 POST /predict 为例：
#   前端发请求：POST /predict，带 JSON { "text": "..." }
#   Uvicorn 收到请求，把它交给 app
#   app 查路由表：发现 POST /predict 对应 predict() 函数
#   app 读取 JSON，用 TextInput 校验并构造对象
#   调用你的 predict() 函数
#   你的 predict() 内部调用 inference.predict_fn(...) 得到结果 dict
#   app 把结果序列化成 JSON 响应，返回给前端
app = FastAPI(
    title="Project 0 Sentiment API",
    description="Sentiment classification API (DistilRoBERTa fine-tuned on IMDb).",
    version="0.1.0"
)

# 全局缓存：模型、tokenizer、device 只在服务启动时加载一次
config = InferenceConfig()
tokenizer = None
model = None
device = None

# ==== 启动事件：加载模型====
@app.on_event("startup")
def load_model():
    global tokenizer, model, device
    print("[startup] Loading model and tokenizer for FastAPI service...")
    tokenizer, model, device = load_model_and_tokenizer(config=config)
    print("[startup] Model and tokenizer loaded.")

# GET = “我要读取/获取信息“：语义是“拿东西“，一般不改变服务器状态
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Sentiment API is running."}

# POST = “我要提交一段数据，请你处理/创建一个结果“：更像“提交表单““发起一个任务“
@app.post("/predict", response_model=PredictOutput)
def predict(input_data: TextInput):
    """
    预测接口：
      输入: {"text": "..."}
      输出: {"label": "...", "score": 0.xx, "probs": {"negative": ..., "positive": ...}}
    """
    if tokenizer is None or model is None or device is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    text = input_data.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    
    try:
        result = predict_fn(text, tokenizer, model, device, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    return PredictOutput(**result)
