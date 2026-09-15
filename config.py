"""公共配置：读 .env + 初始化 LLM/Embedding，供所有 lesson 复用。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 读 .env（项目根目录）
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# 全局 LLM / Embedding
Settings.llm = OpenAILike(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
    max_tokens=2048,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="D:/models/bge-small-zh-v1.5",
    device="cpu",
)

print("[config] LLM 和 Embedding 已加载")