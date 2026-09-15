# LlamaIndex RAG 学习记录

跟一个完整的 RAG 项目学习 LlamaIndex 从入门到独立开发。

## 目标
- 理解 RAG 全链路：文档摄入 → 向量化 → 检索 → 对话生成
- 能独立用 LlamaIndex 写一个 RAG 应用
- 掌握混合检索（向量+BM25）、元数据过滤、多轮对话

## 进度
- [x] Day 1: 环境搭建 + 跑通项目
- [x] Day 2: 核心三件套（ingest / retriever / chat）
- [ ] Day 3: 边角细节
- [ ] Day 4: UI 层
- [ ] Day 5: 动手改功能
- [ ] Day 6: 总结
- [ ] Day 7: 复习

## 练习代码
- ✅ **lesson1**: 最小 RAG（DeepSeek + 本地 bge-small-zh-v1.5）
- ✅ **lesson2**: 手动切分 + 元数据注入 + 持久化
- ✅ **lesson3**: 混合检索（向量 + BM25 + 融合）
- ✅ **lesson4**: 对话引擎（多轮 + 流式 + 引用）
- ✅ **lesson5**: CLI 应用（ingest / ask / chat / list / delete），支持累积入库 + hash 管理

## 参考项目
基于 https://github.com/SocFeng/KnowledgeReview 学习