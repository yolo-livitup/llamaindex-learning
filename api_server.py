"""FastAPI 服务：把 RAG 能力暴露成 HTTP 接口"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import hashlib
import shutil

from llama_index.core import (
    SimpleDirectoryReader, StorageContext, VectorStoreIndex,
    Settings, load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever

import config
from sessions_store import SessionStore


STORAGE_DIR = Path("./storage_lesson2")
SESSIONS_DIR = "./sessions"


# ========== 全局单例 ==========
_index_cache = None
_query_engine_cache = None
_chat_engines = {}   # session_id -> chat_engine
_session_store = SessionStore(SESSIONS_DIR)


def get_index():
    global _index_cache
    if _index_cache is None:
        sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        _index_cache = load_index_from_storage(sc)
    return _index_cache


def get_query_engine():
    global _query_engine_cache
    if _query_engine_cache is not None:
        return _query_engine_cache

    index = get_index()
    vector_retriever = index.as_retriever(similarity_top_k=5)

    docstore = SimpleDocumentStore.from_persist_dir(persist_dir=str(STORAGE_DIR))
    nodes = list(docstore.docs.values())

    if nodes:
        bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25],
            similarity_top_k=5, num_queries=1,
            mode="reciprocal_rerank", use_async=False,
        )
    else:
        retriever = vector_retriever

    _query_engine_cache = RetrieverQueryEngine.from_args(retriever=retriever)
    return _query_engine_cache


def reset_cache():
    """上传/删除文档后调用，让索引重新加载"""
    global _index_cache, _query_engine_cache, _chat_engines
    _index_cache = None
    _query_engine_cache = None
    _chat_engines = {}


# ========== 请求/响应模型 ==========
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict] = []


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    filename: str
    nodes: int
    message: str


# ========== FastAPI 应用 ==========
app = FastAPI(title="LlamaIndex RAG API", version="1.0")


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API 服务运行中"}


@app.get("/health")
def health():
    """健康检查 + 知识库状态"""
    try:
        index = get_index()
        node_count = len(index.docstore.docs)
    except Exception:
        node_count = 0
    return {"status": "ok", "indexed_nodes": node_count}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """单轮提问"""
    engine = get_query_engine()
    response = engine.query(req.question)

    citations = []
    for i, node in enumerate(response.source_nodes, 1):
        citations.append({
            "index": i,
            "file": node.metadata.get("source", "未知"),
            "score": float(node.score) if node.score else 0,
            "text": node.text[:200],
        })

    return QueryResponse(answer=str(response), citations=citations)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """多轮对话（带 session 记忆）"""
    # 新会话就生成 id
    session_id = req.session_id or _session_store.new_id()

    # 加载历史
    history_dicts = _session_store.load(session_id) or []
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in history_dicts
    ]

    # 建/复用 chat_engine
    if session_id not in _chat_engines:
        _chat_engines[session_id] = CondenseQuestionChatEngine.from_defaults(
            llm=Settings.llm,
            query_engine=get_query_engine(),
            chat_history=history,
            verbose=False,
        )

    engine = _chat_engines[session_id]
    response = engine.chat(req.message)

    # 存盘
    history_dicts.append({"role": "user", "content": req.message})
    history_dicts.append({"role": "assistant", "content": str(response)})
    _session_store.save(session_id, history_dicts)

    return ChatResponse(answer=str(response), session_id=session_id)


@app.get("/sessions")
def list_sessions():
    """列出所有会话"""
    return {"sessions": _session_store.list()}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """上传并入库"""
    uploads_dir = STORAGE_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件
    save_path = uploads_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 读 + 切分
    docs = SimpleDirectoryReader(input_files=[str(save_path)]).load_data()
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(docs)

    # 元数据
    file_hash = hashlib.sha1(save_path.read_bytes()).hexdigest()
    for n in nodes:
        n.metadata["source"] = file.filename
        n.metadata["file_hash"] = file_hash
        n.metadata["uploaded_at"] = datetime.now().isoformat()

    # 建 / 追加索引
    if (STORAGE_DIR / "docstore.json").exists():
        index = get_index()
        existing = {n.metadata.get("file_hash") for n in index.docstore.docs.values()}
        if file_hash in existing:
            return UploadResponse(filename=file.filename, nodes=0, message="已存在，跳过")
        for n in nodes:
            index.insert(n)
    else:
        sc = StorageContext.from_defaults()
        index = VectorStoreIndex(nodes, storage_context=sc)

    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    reset_cache()

    return UploadResponse(
        filename=file.filename,
        nodes=len(nodes),
        message=f"入库成功，共 {len(nodes)} 个节点",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)