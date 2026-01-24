# KubAge - Kubernetes知识库系统

KubAge是一个基于Kubernetes文档构建的知识库系统，支持从爬虫、文档解析到向量存储的完整流程。

## 项目结构

```
/workspace/
├── crawler/                 # 爬虫模块
│   ├── K8sCrawler.py        # Kubernetes文档爬虫
│   ├── AsyncDocCrawler.py   # 异步文档爬虫
│   └── doc_crawler.py       # 通用文档爬虫
├── utils/                   # 工具模块
│   ├── MarkdownTreeParser.py # Markdown文档树解析器
│   ├── metadata_utils.py     # 元数据处理工具
│   ├── chunker_utils.py      # 文档切分工具
│   ├── html2md_utils.py      # HTML转Markdown工具
│   └── rag_utils.py          # RAG相关工具
├── workflow/                # 工作流模块
│   └── build_knowledge_base.py # 知识库构建工作流
├── retriever/               # 检索器模块
│   └── MilvusHybridRetriever.py # Milvus混合检索器
├── test/                    # 测试模块
│   ├── build_kb_test.py     # 知识库构建测试
│   └── retrieving_test.py   # 检索功能测试
├── requirements.txt         # 依赖包列表
└── README.md               # 本说明文档
```

## 功能流程

### 1. 爬虫阶段
- **目标**: 从Kubernetes官方文档网站抓取内容
- **工具**: `crawler/K8sCrawler.py`
- **功能**:
  - 异步爬取Kubernetes文档页面
  - 解析HTML内容并转换为Markdown格式
  - 保存为本地`.md`文件
  - 智能去重和错误处理

### 2. Markdown文档存储
- **格式**: Markdown文件（`.md`）
- **路径**: 默认存储在`../raw_data`目录
- **结构**: 按照原始文档结构组织

### 3. 解析Markdown文档
- **工具**: `utils/MarkdownTreeParser.py`
- **功能**:
  - 将Markdown文档解析为树状结构
  - 智能切分文档块，保持语义完整性
  - 提取文档元数据（标题、层级、父子关系等）
  - 处理代码块、表格等特殊内容

### 4. 保存到Milvus
- **工具**: `workflow/build_knowledge_base.py`
- **功能**:
  - 使用密集向量（Qwen）和稀疏向量（BGE-M3）生成双重嵌入表示
  - 将文档数据存储到Milvus向量数据库
  - 支持批量处理和错误恢复

## 数据库结构

Milvus知识库采用混合检索方案，包含以下字段：

| 字段名 | 字段类型 | 说明 |
|--------|----------|------|
| pk | VarChar | 主键 |
| text | VarChar | 文档文本内容 |
| vector | FloatVector | 文本的密集向量表示（Qwen生成） |
| sparse_vector | SparseFloatVector | 文本的稀疏向量表示（BGE-M3生成） |
| title_sparse | SparseFloatVector | 标题的稀疏向量表示（BGE-M3生成） |
| source | VarChar | 源文件路径 |
| title | VarChar | 节点标题 |
| parent_id | VarChar | 父节点ID |
| child_ids | VarChar | 子节点ID列表（JSON格式） |
| node_type | VarChar | 节点类型（ROOT、SECTION、CONTAINER、LEAF） |
| level | Int64 | 节点层级 |
| token_count | Int64 | token数量 |
| left_sibling | VarChar | 左兄弟节点ID |
| right_sibling | VarChar | 右兄弟节点ID |
| from_split | Bool | 是否来自切分 |
| merged | Bool | 是否合并节点 |
| nav_next_step | VarChar | "接下来"章节内容 |
| nav_see_also | VarChar | "另请参见"章节内容 |

### 索引信息

| 索引名称 | 索引类型 | 索引参数 |
|----------|----------|----------|
| vector | FLAT/HNSW | 根据配置决定，metric_type: COSINE |
| sparse_vector | SPARSE_INVERTED_INDEX | metric_type: IP, drop_ratio_build: 0.2 |
| title_sparse | SPARSE_INVERTED_INDEX | metric_type: IP, drop_ratio_build: 0.2 |
| pk | 自动 | 主键索引 |
| source | 二级索引 | 用于快速查找源文档 |
| node_type | 二级索引 | 用于按节点类型筛选 |
| level | 二级索引 | 用于按层级筛选 |
| title | 二级索引 | 用于标题搜索 |

## 检索器介绍

### MilvusHybridRetriever
- **位置**: `retriever/MilvusHybridRetriever.py`
- **功能**:
  - 支持密集向量检索（语义相似性）
  - 支持稀疏向量检索（关键词匹配）
  - 支持标题稀疏向量检索（标题匹配）
  - 使用RRF（Reciprocal Rank Fusion）算法融合多路检索结果
- **特点**: 结合了语义检索和关键词检索的优势，提高了检索准确性

### 检索流程
1. 输入查询文本
2. 使用Qwen模型生成密集查询向量
3. 使用BGE-M3模型生成稀疏查询向量
4. 执行三路检索（文本密集+文本稀疏+标题稀疏）
5. 使用RRF算法融合检索结果
6. 返回融合后的排序结果

## 工作流说明

### 构建知识库工作流
- **位置**: `workflow/build_knowledge_base.py`
- **功能**:
  - 连接Milvus数据库
  - 初始化密集和稀疏嵌入模型
  - 解析Markdown文档为树状结构
  - 生成密集和稀疏向量
  - 批量存入Milvus数据库
  - 创建相应索引
  - 加载集合供查询使用

### 工作流执行步骤
1. **连接数据库**: 建立与Milvus的连接
2. **加载模型**: 初始化Qwen（密集）和BGE-M3（稀疏）嵌入模型
3. **创建Schema**: 定义支持混合检索的数据表结构
4. **解析文档**: 使用MarkdownTreeParser解析文档
5. **生成向量**: 为每份文档生成密集和稀疏向量
6. **批量存储**: 按token数量分批存入Milvus
7. **创建索引**: 为各种向量字段创建相应索引
8. **加载集合**: 将集合加载到内存中供查询使用

## 使用方法

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 爬取文档
```python
from crawler.K8sCrawler import K8sCrawler

# 初始化爬虫
crawler = K8sCrawler(num_workers=5, save_dir="../raw_data")
# 开始爬取
crawler.run()
```

### 3. 构建知识库
```python
from workflow.build_knowledge_base import build_knowledge_base

# 使用默认参数构建知识库
build_knowledge_base(
    embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",
    sparse_model_path="BAAI/bge-m3",  # 稀疏向量模型路径
    markdown_folder_path="../raw_data",
    collection_name="knowledge_base_v2",
    max_tokens_per_batch=2048,
    milvus_host="localhost",
    milvus_port=19530
)
```

### build_knowledge_base函数参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| embedding_model_path | "../models/Qwen/Qwen3-Embedding-0.6B" | 密集嵌入模型路径 |
| sparse_model_path | "BAAI/bge-m3" | 稀疏嵌入模型路径 |
| markdown_folder_path | "../raw_data" | Markdown文件夹路径 |
| collection_name | "knowledge_base_v2" | Milvus集合名 |
| max_tokens_per_batch | 2048 | 批量存入数据库的批token数量上限 |
| min_chunk_size | 256 | 最小块大小 |
| core_chunk_size | 512 | 核心块大小 |
| max_chunk_size | 2048 | 最大块大小 |
| milvus_host | 从环境变量读取 | Milvus主机地址 |
| milvus_port | 从环境变量读取 | Milvus端口 |
| milvus_user | 从环境变量读取 | Milvus用户名 |
| milvus_password | 从环境变量读取 | Milvus密码 |
| index_type | "FLAT" | 索引类型 |
| metric_type | "COSINE" | 度量类型 |

### 4. 使用检索器
```python
from test.retrieving_test import main

# 运行检索测试
main()
```

## 配置文件

项目使用`.env`文件存储配置信息：

```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_ROOT_PASSWORD=your_password
```

## 数据处理流程

1. **爬虫阶段**: 从Kubernetes官网爬取HTML文档
2. **格式转换**: 将HTML转换为Markdown格式
3. **文档解析**: 解析Markdown为树状结构
4. **内容切分**: 按语义和大小切分文档块
5. **密集向量化**: 使用Qwen模型生成密集向量表示
6. **稀疏向量化**: 使用BGE-M3模型生成稀疏向量表示
7. **存储**: 将向量和元数据存储到Milvus
8. **索引**: 为向量和元数据建立索引
9. **检索**: 使用混合检索器进行查询

## Agent架构

### Agent概述

KubAge新增了智能Agent模块，用于理解和分析用户在Kubernetes运维场景中的问题，并提供智能化的解决方案。Agent通过多轮对话理解用户意图，识别操作风险，并生成相应的检索Query来查询知识库。

### Agent状态结构 (AgentState)

Agent使用TypedDict定义状态结构，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| messages | List[BaseMessage] | 消息历史记录，使用operator.add实现追加模式 |
| analysis | ProblemAnalysis \| None | 存储问题分析结果 |
| retrieved_chunks | List[Document] \| None | 检索到的文档片段 |
| generated_response | str \| None | 生成的响应结果 |
| metadata | Dict[str, Any] \| None | 元数据信息 |
| error | str \| None | 错误信息 |

### Agent节点 (Nodes)

Agent包含以下几种节点类型，每个节点负责不同的处理任务：

#### 1. Analysis Node（分析节点）
- **功能**: 对用户输入进行深度分析，提取关键信息
- **输入**: 用户消息历史和当前输入
- **输出**: 结构化的ProblemAnalysis对象
- **处理流程**:
  1. 使用JsonOutputParser解析ProblemAnalysis模型
  2. 通过LLM进行意图识别、实体提取和风险评估
  3. 生成检索Query列表
  4. 输出包含思维链推理的结构化分析结果

#### 2. Retrieval Node（检索节点）
- **功能**: 基于分析结果中的搜索查询，从Milvus知识库中检索相关文档
- **输入**: 分析结果中的search_queries
- **输出**: 检索到的相关文档片段列表
- **处理流程**:
  1. 获取上一节点的分析结果
  2. 遍历所有搜索查询进行检索
  3. 对检索结果进行去重处理
  4. 返回唯一文档片段列表

#### 3. Rerank Node（重排序节点）
- **功能**: 对检索到的文档进行重排序，提高相关性
- **输入**: 检索到的文档片段列表
- **输出**: 重新排序后的文档片段列表
- **处理流程**:
  1. 基于操作类型生成动态指令
  2. 使用Qwen3-Reranker模型计算文档相关性分数
  3. 根据分数重新排序文档
  4. 返回Top-N最相关文档

### 问题分析结构 (ProblemAnalysis)

ProblemAnalysis是Pydantic模型，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 思维链推理过程：上下文消歧、意图分析、信息提取、风险判断 |
| technical_summary | str | 用户问题的技术摘要，去除口语化表达 |
| target_operation | OperationType | 目标操作类型 |
| entities | List[NamedEntity] | 提取的关键命名实体列表 |
| risk_level | RiskLevel | 操作风险等级 |
| search_queries | List[str] | 用于知识库检索的Query列表 |
| clarification_question | str \| None | 如果信息不足需要追问的问题 |

### 操作类型 (OperationType)

| 枚举值 | 说明 |
|--------|------|
| KNOWLEDGE_QA | 知识问答 |
| RESOURCE_CREATION | 资源创建 |
| RESOURCE_MUTATION | 资源变更 |
| RESOURCE_DELETION | 资源删除 |
| RESOURCE_INQUIRY | 资源查询 |
| DIAGNOSIS | 故障诊断 |
| CONFIGURE | 配置变更 |
| SCALING | 性能调优——水平伸缩 |
| RESTART | 重启运行时 |
| ROLLOUT | 回滚 |
| PROXY | 代理 |
| OTHER | 其他操作 |

### 风险等级 (RiskLevel)

| 枚举值 | 说明 |
|--------|------|
| LOW | 信息查询，只读操作 |
| MEDIUM | 配置检查、非破坏性调试 |
| HIGH | 配置变更、重启 |
| CRITICAL | 删除资源、危险操作 |

### Agent工作流模式

Agent采用LangGraph构建有状态的工作流，支持以下模式：

1. **顺序执行模式**: 按照分析→检索→重排序的顺序执行
2. **条件分支模式**: 根据分析结果决定后续处理路径
3. **循环重试模式**: 在检索或重排序失败时进行重试

### Agent节点测试和测试用例

Agent的功能通过多种测试用例进行全面验证：

#### 分析节点测试 (Analysis Node Tests)
- **上下文实体提取**: 测试从多轮对话中提取实体的能力
- **指代消歧**: 测试识别代词指代的具体资源（如"它"指代具体资源）
- **意图识别**: 测试准确识别用户操作意图的能力
- **风险评估**: 测试正确评估操作风险等级的能力
- **检索Query生成**: 测试生成有效的知识库检索Query

#### 检索节点测试 (Retrieval Node Tests)
- **多Query检索**: 测试同时使用多个查询进行检索
- **文档去重**: 测试对重复文档进行去重处理
- **错误处理**: 测试单个查询失败时的容错机制

#### 重排序节点测试 (Rerank Node Tests)
- **相关性评分**: 测试使用Qwen3-Reranker模型计算文档相关性
- **动态指令**: 测试根据不同操作类型生成针对性指令
- **批处理推理**: 测试高效处理大量文档对的能力

#### 具体测试场景包括：
- **故障诊断场景**: 如"redis-cart连接问题"的上下文分析
- **跨命名空间分析**: 处理涉及多个命名空间的复杂问题
- **高风险操作识别**: 如删除操作的危险性识别
- **水平伸缩操作**: 处理Deployment扩容等伸缩需求
- **知识问答**: 纯理论问题的处理
- **回滚操作**: 版本回滚等重大变更操作
- **资源创建**: 新建Namespace等资源创建操作
- **低风险查询**: 日志查看等只读操作

### Agent工作流示例

```python
from langgraph.graph import StateGraph
from agent.nodes.analysis_node import AnalysisNode
from agent.nodes.retrieval_node import RetrievalNode
from agent.nodes.rerank_node import RerankNode
from agent.state import AgentState
from utils.llm_factory import get_chat_model

# 构建工作流
workflow = StateGraph(AgentState)

# 初始化各节点
llm = get_chat_model()
analysis_node = AnalysisNode(llm)
retrieval_node = RetrievalNode(retriever)
rerank_node = RerankNode(model_path="../models/Qwen/Qwen3-Reranker-0.6B")

# 添加节点
workflow.add_node("analyze_problem", analysis_node)
workflow.add_node("retrieve_docs", retrieval_node)
workflow.add_node("rerank_docs", rerank_node)

# 定义边
workflow.add_edge(START, "analyze_problem")
workflow.add_edge("analyze_problem", "retrieve_docs")
workflow.add_edge("retrieve_docs", "rerank_docs")
workflow.add_edge("rerank_docs", END)

app = workflow.compile()

# 运行工作流
result = app.invoke({"messages": [HumanMessage(content="帮我排查nginx部署的连接问题")]})
```

## 依赖包

主要依赖包包括：
- `langchain-huggingface`: HuggingFace嵌入模型集成
- `langchain-milvus`: Milvus向量数据库集成
- `transformers`: 模型和分词器
- `torch`: PyTorch深度学习框架
- `pymilvus`: Milvus Python SDK
- `pymilvus[model]`: Milvus模型库（包含BGE-M3等）
- `beautifulsoup4`: HTML解析
- `python-dotenv`: 环境变量管理
- `langgraph`: Agent工作流编排
- `pydantic`: 数据验证和设置管理