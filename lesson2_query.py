"""lesson2: 从磁盘加载索引并提问"""
from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv(override=True)

Settings.llm = OpenAILike(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
    max_tokens=2048,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="D:/models/bge-small-zh-v1.5",
    device="cpu",
)

storage_context = StorageContext.from_defaults(persist_dir="./storage_lesson2")
index = load_index_from_storage(storage_context)

response = index.as_query_engine().query("这个文档讲了什么？")
print(response)