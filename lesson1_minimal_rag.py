# 文件地址 "D:\P-python-code\langchain1.2_learning\asset\load\10-test_doc.txt"
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


load_dotenv(override=True)
deep_seek_key = os.getenv('DEEPSEEK_API_KEY')
deep_seek_model_name = os.getenv('LLM_MODEL')

Settings.llm = OpenAILike(
    model=deep_seek_model_name,
    api_key=deep_seek_key,
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
    max_tokens=2048,
)

documents = SimpleDirectoryReader(input_files=[r"D:\P-python-code\langchain1.2_learning\asset\load\10-test_doc.txt"]).load_data()



Settings.embed_model = HuggingFaceEmbedding(
    model_name="D:/models/bge-small-zh-v1.5",
    device="cpu",
)

index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("这个文档讲了什么？")
print(response)
