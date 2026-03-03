🌟 一、 目前的优势与亮点（做得好的地方）
极其优秀的模块化解耦 (core)
摄入 (Ingestion)、检索 (Retrieval)、推理提取 (Reasoning)、图谱 (KG) 和本体 (Ontology) 各司其职。这种设计使得未来将向量库替换为 Milvus，或将本地 JSON 图谱替换为 Neo4j 都变得轻而易举，无需重构核心业务。
强类型与 Schema 约束 (schemas)
强制要求使用 Pydantic 定义 PaperExtraction 等格式。这在不可控的 LLM 开发中是关键的护城河，为后续所有数据的可视化提供了保障。
“离线假数据”模式 (dummy.py)
内建了 Dummy 甚至 Mock Retriever，这是一个极其优秀的工程习惯（防御性编程），使得可以在不消耗昂贵 API Token 的情况下跑通流水线、调试 UI 和图谱逻辑。
增量更新能力（刚刚引入）
已经可以按需跳过已分析的论文并动态融合子图（Subgraph），这为构建一个长期的“个人/团队学术知识库”奠定了物理基础。

⚠️ 二、 核心痛点与技术债（亟需解决的不足）
尽管架构优美，但在真正的“硬核科研”场景下，它的具体实现还偏向薄弱：

1. Ingestion (数据摄入) 层的模态丢失
现状：目前通常只是把 PDF 或纯文本进行暴力的 Text 提取（loader.py -> chunker.py）。
痛点：对于学术论文，图表、对比表格、数学公式往往包含了最核心的 Methods 和 Metrics。暴力文本提取会把复杂的表格揉成乱码，导致 LLM 完全无法提取出准确的对比数据。
反思：需要引入专门的学术解析器（如 Nougat, Marker, 或多模态模型直接看图）。

2. Ontology (本体/实体) 的消岐能力太弱
现状：在 kg/builder.py 中，生成实体 ID 只是简单地转小写、去标点（clean_name.strip('_') 等）。
痛点：由于大模型的发散性，同一篇论文可能这次提取出方法叫 LLM-based Generation，另一篇叫 Large Language Model Generation。按目前的逻辑，它们在知识图谱里会变成两个毫无关联的孤立节点（Node）。
反思：在生成图谱前，亟需一个实体共指消解 (Entity Resolution / Co-reference) 步骤。这是任何实用型知识图谱必须跨越的门槛（需要利用向量相似计算来进行实体合并）。

3. Reasoning (推理) 层的 JSON 脆弱性
现状：ReasoningEngine 严重依赖大模型一次性输出完美的、符合复杂嵌套 Schema 的 json_object。
痛点：如果接入我们之前讨论的 Minimind 或其他较小的本地模型，大概率会因为 JSON 格式错乱（漏了逗号、嵌套错误）、或者幻觉输出了未定义的字段而导致整个 Pipeline 崩溃（Crash）。
反思：缺少 “重试机制 (Retry)” 和 “结构化纠错 (Fallback/Repair)”。应该引入类似 Instructor 或 Outlines 的约束生成，或者在遇到 JSON 解析失败时给 LLM 报错并要求其重新生成。

4. Retrieval (检索) 还未真正发挥图谱优势
现状：从 Orchestrator 的 answer_question 方法看，目前依然是使用传统的 VectorRetriever 找 Text Chunk，然后塞给 LLM。辛苦建好的知识图谱 (KnowledgeGraph) 只是用来做 Streamlit 可视化的。
痛点：这是一个“伪 GraphRAG”。真正的 GraphRAG 应该在用户提问时，沿着知识图谱的节点（Nodes）和边（Edges）进行游走和检索，提供跨论文的逻辑链条。



3. 迭代 RAG + KG (多步推理管道设计)

契合度：高。 现有的 qa.py （问答助手）很可能只是把检索到的文本块 chunk 和图谱上下文一股脑塞给 LLM（Single-pass RAG）。这在回答“A论文和B论文在解决长尾分布问题上的核心分歧是什么”这种复杂逻辑时往往会失败。
如何优化进项目：
在 reasoning 模块中引入类似 ReAct 或 LangChain Agent 的机制。
给予 ResearchAssistant 几个工具（Functions/Tools）：search_kg(entity)（查图谱节点）、search_vector(query)（查原文细节）、compare_nodes(node_a, node_b)。
让 LLM 自己决定需要分几步来查阅资料，然后再给出最终综述。这种 Iterative (迭代) 的方式是当前学术 RAG 的标配。

gliner实体提取+llm关系提取加实体过滤+向量匹配去重

neo4j