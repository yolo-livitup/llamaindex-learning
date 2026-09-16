"""Streamlit 网页版 RAG 工具：文档管理 + 对话"""
import streamlit as st
from pathlib import Path
from datetime import datetime
import hashlib

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
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


# ========== 缓存：全局只初始化一次 ==========
@st.cache_resource
def get_session_store():
    return SessionStore(SESSIONS_DIR)


@st.cache_resource
def load_index_cached():
    sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(sc)


def build_retriever(index):
    vector_retriever = index.as_retriever(similarity_top_k=5)
    docstore = SimpleDocumentStore.from_persist_dir(persist_dir=str(STORAGE_DIR))
    nodes = list(docstore.docs.values())
    if not nodes:
        return vector_retriever
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


@st.cache_resource
def get_query_engine():
    index = load_index_cached()
    retriever = build_retriever(index)
    return RetrieverQueryEngine.from_args(retriever=retriever)


# ========== 侧边栏：文档管理 ==========
def render_sidebar():
    with st.sidebar:
        st.title("📚 RAG 知识库")
        st.divider()

        st.subheader("📤 上传文档")
        uploaded = st.file_uploader(
            "选择文件", type=["txt", "md"], accept_multiple_files=True
        )
        if uploaded and st.button("开始入库"):
            _ingest_files(uploaded)

        st.divider()

        st.subheader("📂 已入库文档")
        _render_doc_list()


def _ingest_files(uploaded_files):
    storage_dir = STORAGE_DIR
    storage_dir.mkdir(exist_ok=True)
    uploads_dir = storage_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    total_nodes = 0
    for f in uploaded_files:
        # 保存文件
        path = uploads_dir / f.name
        path.write_bytes(f.getbuffer().tobytes())

        # 读文件
        docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
        nodes = splitter.get_nodes_from_documents(docs)

        # 元数据
        file_hash = hashlib.sha1(path.read_bytes()).hexdigest()
        for n in nodes:
            n.metadata["source"] = f.name
            n.metadata["file_hash"] = file_hash
            n.metadata["uploaded_at"] = datetime.now().isoformat()

        # 建/追加索引
        if (storage_dir / "docstore.json").exists():
            index = load_index_cached()
            existing = {n.metadata.get("file_hash") for n in index.docstore.docs.values()}
            if file_hash in existing:
                st.warning(f"⏭️ {f.name} 已存在，跳过")
                continue
            for n in nodes:
                index.insert(n)
        else:
            sc = StorageContext.from_defaults()
            index = VectorStoreIndex(nodes, storage_context=sc)

        index.storage_context.persist(persist_dir=str(storage_dir))
        total_nodes += len(nodes)
        st.success(f"✅ {f.name}：{len(nodes)} 个节点")

    # 清缓存，让下次查询用新索引
    load_index_cached.clear()
    get_query_engine.clear()
    st.success(f"🎉 入库完成，共 {total_nodes} 个节点")


def _render_doc_list():
    if not (STORAGE_DIR / "docstore.json").exists():
        st.caption("暂无文档")
        return

    docstore = SimpleDocumentStore.from_persist_dir(persist_dir=str(STORAGE_DIR))
    groups = {}
    for n in docstore.docs.values():
        h = n.metadata.get("file_hash", "unknown")
        groups.setdefault(h, []).append(n)

    for h, nodes in groups.items():
        source = nodes[0].metadata.get("source", "?")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"📄 {source}  ({len(nodes)} 节点)")
        with col2:
            if st.button("🗑️", key=f"del_{h}"):
                _delete_doc(h)
                st.rerun()


def _delete_doc(file_hash: str):
    index = load_index_cached()
    to_delete = [
        n.node_id
        for n in index.docstore.docs.values()
        if n.metadata.get("file_hash", "").startswith(file_hash)
    ]
    for node_id in to_delete:
        index.docstore.delete_document(node_id, raise_error=False)
    try:
        index.vector_store.delete_nodes(to_delete)
    except Exception:
        pass
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    load_index_cached.clear()
    get_query_engine.clear()
    st.toast(f"✅ 已删除 {len(to_delete)} 个节点")


# ========== 主界面：对话 ==========
def render_chat():
    st.title("💬 知识问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # 历史消息
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                with st.expander(f"📎 引用 {len(m['citations'])} 个片段"):
                    for c in m["citations"]:
                        st.caption(f"[{c['index']}] {c['file']}  (score={c['score']:.4f})")
                        st.text(c["text"][:150] + "...")

    # 输入
    user_input = st.chat_input("请输入问题…")
    if not user_input:
        return

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input, "citations": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 建 chat_engine（只建一次）
    if st.session_state.chat_engine is None:
        qe = get_query_engine()
        st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
            llm=Settings.llm,
            query_engine=qe,
            verbose=False,
        )

    # 流式回答
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        r = st.session_state.chat_engine.stream_chat(user_input)
        for token in r.response_gen:
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

        # 引用
        citations = []
        if hasattr(r, "source_nodes") and r.source_nodes:
            with st.expander(f"📎 引用 {len(r.source_nodes)} 个片段"):
                for i, node in enumerate(r.source_nodes, 1):
                    fname = node.metadata.get("source", "未知")
                    score = node.score or 0
                    st.caption(f"[{i}] {fname}  (score={score:.4f})")
                    st.text(node.text[:150] + "...")
                    citations.append({
                        "index": i,
                        "file": fname,
                        "score": score,
                        "text": node.text,
                    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "citations": citations,
    })


# ========== 主入口 ==========
def main():
    st.set_page_config(page_title="RAG 知识库", page_icon="📚", layout="wide")
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()