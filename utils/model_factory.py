import os

import torch
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')

for var in [API_KEY, BASE_URL, MODEL_NAME]:
    if var is None:
        raise ValueError(f"环境变量缺失：{['API_KEY', 'BASE_URL', 'MODEL_NAME'][[API_KEY, BASE_URL, MODEL_NAME].index(var)]}")

os.environ["LANGCHAIN_TRACING_V2"] = "false"


def get_chat_model(temperature=0, max_token=16384, frequency_penalty=0, top_p=0.95, extra_body=None) -> ChatOpenAI:
    """
    获取配置好的 ChatModel 实例
    """
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=max_token,
        frequency_penalty=frequency_penalty,
        top_p=top_p,
        extra_body=extra_body
    )

def get_dense_embed_model(model_path):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
            # "use_flash_attention_2": True
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

def get_sparse_embed_model(model_path):
    return BGEM3EmbeddingFunction(
        model_name=model_path,
        use_fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )