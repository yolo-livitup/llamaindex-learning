import hashlib
from datetime import datetime
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex , Settings
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
import config


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


class RagApp:
    def __init__(self):
        #索引
        self.index = load_index()

        #混合检索
        vector_retriever = self.index.as_retriever(similarity_top_k=5)
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir="./storage_lesson2")
        nodes = list(docstore.docs.values())
        b25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, b25_retriever],
            similarity_top_k=5,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False
        )

        self.query_engine = RetrieverQueryEngine.from_args(retriever=self.fusion_retriever)

    def cmd_ask(self,question):

        answer = self.query_engine.query(question)
        print(answer)
        print("\n📎 引用：")
        for i, node in enumerate(answer.source_nodes, 1):
            fname = node.metadata.get("source", "未知")
            print(f"  [{i}] {fname}")
            print(f"      {node.text[:80]}...")




    def cmd_chat(self):
        my_chat_engine = CondenseQuestionChatEngine.from_defaults(
            llm=Settings.llm,
            query_engine=self.query_engine,
            verbose=True,
        )
        while True:
            user_input = input("你：")
            if user_input in ("quit", "exit"):
                break

            r = my_chat_engine.stream_chat(user_input)
            print("AI: ", end="", flush=True)
            for token in r.response_gen:
                print(token, end="", flush=True)
            print()

import argparse

def main():
    parser = argparse.ArgumentParser(description="LlamaIndex RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 子命令 1: ingest
    p_ingest = sub.add_parser("ingest", help="上传文件并入库")
    p_ingest.add_argument("file", help="要入库的文件路径")

    # 子命令 2: ask
    p_ask = sub.add_parser("ask", help="单轮提问")
    p_ask.add_argument("question", help="要问的问题")

    # 子命令 3: chat
    sub.add_parser("chat", help="多轮对话模式")

    args = parser.parse_args()

    if args.cmd == "ingest":
        cmd_ingest(args.file)
    elif args.cmd == "ask":
        app = RagApp()
        app.cmd_ask(args.question)
    elif args.cmd == "chat":
        app = RagApp()
        app.cmd_chat()


if __name__ == "__main__":
    main()
