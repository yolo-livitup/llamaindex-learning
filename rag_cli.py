import hashlib
from datetime import datetime
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex , Settings
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
import config
import contextvars
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from sessions_store import SessionStore


def load_index():
    from llama_index.core import StorageContext, load_index_from_storage
    sc = StorageContext.from_defaults(persist_dir="./storage_lesson2")
    return load_index_from_storage(sc)


def cmd_ingest(file:str):
    #数据加载
    path_file = Path(file)
    documents = SimpleDirectoryReader(input_files=[path_file]).load_data()

    #手动构建切分器,拿到节点（nodes）
    splitter = SentenceSplitter(chunk_size=256,chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(documents)

    #手动添加元数据
    upload_time = datetime.now().isoformat()
    file_hash = hashlib.sha1(path_file.read_bytes()).hexdigest()
    for node in nodes:
        node.metadata["source"] = path_file.name
        node.metadata["file_hash"] = file_hash
        node.metadata["uploaded_at"] = upload_time

    #建造仓库
    storage_dir = Path("./storage_lesson2")

    if (storage_dir / "docstore.json").exists():
        index = load_index()

        # ⭐ 新增：检查 hash 是否已存在
        existing_hashes = {
            n.metadata.get("file_hash")
            for n in index.docstore.docs.values()
        }
        if file_hash in existing_hashes:
            print(f"⏭️ 文件已存在（hash={file_hash[:10]}），跳过")
            return

        # 追加新节点
        for n in nodes:
            index.insert(n)
        print(f"➕ 追加 {len(nodes)} 个节点")
    else:
        storage_dir.mkdir(exist_ok=True)
        sc = StorageContext.from_defaults()
        index = VectorStoreIndex(nodes, storage_context=sc)
        print(f"🆕 首次建索引，{len(nodes)} 个节点")

        # 持久化
    index.storage_context.persist(persist_dir=str(storage_dir))
    print(f"✅ 入库完成，累计 {len(index.docstore.docs)} 个节点")

# ========== 按文件过滤的检索器 ==========
_doc_filter_var: contextvars.ContextVar = contextvars.ContextVar(
    "doc_filter", default=None
)


def set_doc_filter(file_names):
    """设置本次请求只在哪些文件里检索"""
    return _doc_filter_var.set(set(file_names) if file_names else None)


def reset_doc_filter(token):
    """用完还原"""
    _doc_filter_var.reset(token)


class FilteredRetriever(BaseRetriever):
    """包一层：在内层检索后，按当前请求的 doc_filter 过滤结果"""

    def __init__(self, base_retriever):
        super().__init__()
        self._base = base_retriever

    def _retrieve(self, query_bundle: QueryBundle):
        nodes = self._base.retrieve(query_bundle)
        allowed = _doc_filter_var.get()
        if allowed:
            nodes = [
                n for n in nodes
                if n.node.metadata.get("source") in allowed
            ]
        return nodes








class RagApp:
    def __init__(self):
        #索引
        self.index = load_index()
        self.session_store = SessionStore()
        #混合检索
        vector_retriever = self.index.as_retriever(similarity_top_k=5)
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir="./storage_lesson2")
        nodes = list(docstore.docs.values())
        if nodes:
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
            fusion = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=5,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
            # 包一层过滤
            self.fusion_retriever = FilteredRetriever(fusion)
        else:
            print("⚠️ 索引为空，仅使用向量检索")
            self.fusion_retriever = FilteredRetriever(vector_retriever)

        self.query_engine = RetrieverQueryEngine.from_args(retriever=self.fusion_retriever)

    def cmd_ask(self, question: str, files: list = None):
        token = set_doc_filter(files)  # 设置过滤
        try:
            answer = self.query_engine.query(question)
            print("AI:", answer)
            print("\n📎 引用：")
            for i, node in enumerate(answer.source_nodes, 1):
                fname = node.metadata.get("source", "未知")
                print(f"  [{i}] {fname}")
                print(f"      {node.text[:80]}...")
        finally:
            reset_doc_filter(token)  # 还原

    def cmd_chat(self, session_id=None, files: list = None):
        if not session_id:
            session_id = self.session_store.new_id()
            print(f"🆕 新会话：{session_id}")
        else:
            print(f"📂 继续会话：{session_id}")

        history = self.session_store.load(session_id) or []
        messages = [
            ChatMessage(role=h["role"], content=h["content"])
            for h in history
        ]

        my_chat_engine = CondenseQuestionChatEngine.from_defaults(
            llm=Settings.llm,
            query_engine=self.query_engine,
            verbose=True,
            chat_history=messages,
        )

        print(f"📁 检索范围：{files or '全部'}")
        print("输入 quit 退出\n")

        while True:
            user_input = input("你：")
            if user_input in ("quit", "exit"):
                break

            token = set_doc_filter(files)
            try:
                r = my_chat_engine.stream_chat(user_input)
                print("AI: ", end="", flush=True)
                collected = []
                for tok in r.response_gen:
                    print(tok, end="", flush=True)
                    collected.append(tok)
                print()
            finally:
                reset_doc_filter(token)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": "".join(collected)})
            self.session_store.save(session_id, history)

        print(f"✅ 会话已保存：{session_id}")
    def cmd_list(self):
        """列出所有文档（按 file_hash 分组）"""
        groups ={}
        nodes = self.index.docstore.docs.values()
        for n in nodes:
            h = n.metadata["file_hash"]
            groups.setdefault(h, []).append(n)

        print(f"共 {len(groups)} 个文档，{sum(len(v) for v in groups.values())} 个节点")
        for h,nodes in groups.items():
            source = nodes[0].metadata.get("source","?")
            print(f"  [{h[:10]}]  {source}  ({len(nodes)} 个节点)")

    def cmd_delete(self, file_hash: str):
        """删除指定 hash 的所有节点"""
        to_delete =[]
        matched_hash = None
        nodes = self.index.docstore.docs.values()
        for n in nodes:
            full_h = n.metadata["file_hash"]
            if full_h== file_hash or full_h.startswith(file_hash):
                to_delete.append(n.node_id)
                matched_hash = full_h
        #未找到
        if not to_delete:
            print(f"❌ 没找到 hash={file_hash} 的文档")
            return

        for n in to_delete:
            self.index.docstore.delete_document(n,raise_error=False)


        try:
            self.index.vector_store.delete_nodes(to_delete)
        except Exception:
            pass


        # ④ 持久化
        self.index.storage_context.persist(persist_dir="./storage_lesson2")
        print(f"✅ 已删除 {len(to_delete)} 个节点  (hash={matched_hash[:10]}...)")




import argparse

def main():
    parser = argparse.ArgumentParser(description="LlamaIndex RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("file")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--files", default=None,
                       help="只在指定文件中检索，逗号分隔")

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("session_id", nargs="?", default=None)
    p_chat.add_argument("--files", default=None,
                        help="只在指定文件中检索，逗号分隔")

    sub.add_parser("list")

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("file_hash")

    args = parser.parse_args()

    # 解析 --files（逗号分隔 → 列表）
    files = None
    if hasattr(args, "files") and args.files:
        files = [f.strip() for f in args.files.split(",")]

    if args.cmd == "ingest":
        cmd_ingest(args.file)
    elif args.cmd == "ask":
        app = RagApp()
        app.cmd_ask(args.question, files=files)
    elif args.cmd == "chat":
        app = RagApp()
        app.cmd_chat(args.session_id, files=files)
    elif args.cmd == "list":
        app = RagApp()
        app.cmd_list()
    elif args.cmd == "delete":
        app = RagApp()
        app.cmd_delete(args.file_hash)


if __name__ == "__main__":
    main()
