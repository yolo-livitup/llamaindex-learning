from llama_index.core import load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from lesson2_ingest import storage_context


index = load_index_from_storage(storage_context)

#向量检索器
vector_retriever = index.as_retriever(similarity_top_k=5)

#拿原始文本,在建立B25检索器
nodes= list(index.docstore.docs.values())
b25_retriever = BM25Retriever.from_defaults(nodes=nodes,similarity_top_k=5)


#两个检索其融合
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever,b25_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False
)


from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(retriever=fusion_retriever)
print(query_engine.query("我想去比较休闲的地方，给我几个推荐"))