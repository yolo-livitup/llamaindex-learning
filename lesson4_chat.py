from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import config
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine

# 加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage_lesson2")
index = load_index_from_storage(storage_context)

# 检索器
vector_retriever = index.as_retriever(similarity_top_k=5)
nodes = list(index.docstore.docs.values())

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)

query_engine = RetrieverQueryEngine.from_args(retriever=fusion_retriever)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    llm=Settings.llm,
    verbose=True,
)

# ========== 5. 流式对话 + 引用 ==========
print("\n" + "=" * 60)
print("第 1 轮（流式）")
print("=" * 60)

r1 = chat_engine.stream_chat("北京故宫怎么样？")
print("AI: ", end="", flush=True)
for token in r1.response_gen:
    print(token, end="", flush=True)
print()

# 打印引用
print("\n📎 引用：")
for i, node in enumerate(r1.source_nodes, 1):
    file_name = node.metadata.get("source", "未知")
    score = f"{node.score:.4f}" if node.score else "-"
    preview = node.text[:60].replace("\n", " ")
    print(f"  [{i}] {file_name} (score={score})")
    print(f"      {preview}...")

# ========== 第 2 轮（测试多轮）==========
print("\n" + "=" * 60)
print("第 2 轮（测试多轮记忆）")
print("=" * 60)

r2 = chat_engine.stream_chat("它有多大？")
print("AI: ", end="", flush=True)
for token in r2.response_gen:
    print(token, end="", flush=True)
print()