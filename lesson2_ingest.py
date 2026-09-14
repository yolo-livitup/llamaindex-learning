"""lesson2: 手动切分 + 元数据注入 + 持久化"""
import hashlib
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
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
#文件数据加载
FILE_PATH = Path(r"D:\P-python-code\langchain1.2_learning\asset\load\10-test_doc.txt")
documents = SimpleDirectoryReader(input_files=[FILE_PATH]).load_data()

# 建立切分器，切分documents -> nodes
splitter = SentenceSplitter(chunk_size=100,chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

# 元数据注入,准备文件夹
file_hash = hashlib.sha1(FILE_PATH.read_bytes()).hexdigest()
uploaded_at = datetime.now().isoformat()

for n in nodes:
    n.metadata["source"] = FILE_PATH.name
    n.metadata["file_hash"] = file_hash
    n.metadata["uploaded_at"] = uploaded_at


# 建文件和存放位置
storage_dir = Path("./storage_lesson2")
storage_dir.mkdir(exist_ok=True)

storage_context = StorageContext.from_defaults()
index = VectorStoreIndex(nodes,storage_context=storage_context)

#磁盘持久化
index.storage_context.persist(persist_dir=str(storage_dir))
print(f"索引已持久化到 {storage_dir}")




