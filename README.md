# KubAge - Kubernetes知识库系统

KubAge是一个基于Kubernetes文档构建的智能知识库系统，支持从文档爬取、解析、向量存储到智能问答的完整流程。系统采用现代化的Agent架构，结合混合检索技术和MCP协议，为用户提供专业的Kubernetes运维支持。

## 项目结构

```
/workspace/
├── crawler/                 # 文档爬虫模块
│   ├── K8sCrawler.py        # Kubernetes文档爬虫
│   ├── AsyncDocCrawler.py   # 异步文档爬虫
│   ├── doc_crawler.py       # 通用文档爬虫
│   └── crawler_analysis.py  # 爬虫分析工具
├── utils/                   # 工具模块
│   ├── MarkdownTreeParser.py # Markdown文档树解析器
│   ├── document_schema.py   # 文档数据结构定义
│   ├── chunker_utils.py     # 文档切分工具
│   ├── html2md_utils.py     # HTML转Markdown工具
│   ├── rag_utils.py         # RAG相关工具
│   ├── milvus_adapter.py    # Milvus适配器
│   ├── model_factory.py     # 模型工厂
│   └── mcp_manager.py       # MCP管理器
├── workflow/                # 工作流模块
│   └── build_knowledge_base.py # 知识库构建工作流
├── retriever/               # 检索器模块
│   ├── MilvusHybridRetriever.py # Milvus混合检索器
│   └── GraphTraverser.py    # 图遍历检索器
├── agent/                   # Agent智能体模块
│   ├── nodes/               # Agent节点
│   │   ├── sensory_node.py     # 感知节点
│   │   ├── analysis_node.py    # 问题分析节点
│   │   ├── planning_node.py    # 规划节点
│   │   ├── retrieval_node.py   # 文档检索节点
│   │   ├── tool_node.py        # 工具调用节点
│   │   ├── regulation_node.py  # 自我调节节点
│   │   ├── expression_node.py  # 表达节点
│   │   └── rerank_node.py      # 重排序节点
│   ├── graph.py             # Agent工作流图
│   ├── state.py             # Agent状态定义
│   ├── schemas.py           # 数据结构定义
│   └── prompts.py           # Prompt模板
├── informer/                # 信息桥接模块
│   └── RuntimeBridge.py     # 运行时桥接器
├── os_mcp/                  # 操作系统MCP服务
│   └── os_mcp_server.py     # MCP服务器实现
├── config/                  # 配置模块
│   └── mcp_config.json      # MCP服务配置
├── init-scripts/            # 初始化脚本
│   └── 001-enable-pgvector.sql # PostgreSQL向量扩展初始化脚本
├── test/                    # 测试模块
│   ├── agent_node_test.py      # Agent节点测试
│   ├── build_kb_test.py        # 知识库构建测试
│   ├── milvus_test.py          # Milvus测试
│   ├── mcp_client_test.py      # MCP客户端测试
│   ├── torch_test.py           # PyTorch测试
│   ├── re_test.py              # 正则表达式测试
│   ├── retrieval_node_test.py  # 检索节点测试
│   ├── retrieving_test.py      # 检索功能测试
│   ├── visualize_test.py       # 可视化测试
│   ├── flash-attn_test.py      # Flash Attention测试
│   └── preprocess_api_blocks_test.py # API块预处理测试
├── docker-compose-milvus.yaml # Milvus Docker Compose配置
├── docker-compose-pg.yaml     # PostgreSQL Docker Compose配置
├── main.py                    # 主程序入口
├── baseline.py                # 基线测试程序
├── init_postgres.py           # PostgreSQL初始化脚本
├── milvus_build.bat           # Milvus构建批处理脚本
├── milvus_clean.bat           # Milvus清理批处理脚本
├── pg_build.bat               # PostgreSQL构建批处理脚本
├── pg_clean.bat               # PostgreSQL清理批处理脚本
├── requirements.txt           # 依赖包列表
└── README.md                  # 本说明文档
```

## 核心功能模块

### 1. 爬虫模块 (crawler/)
- **K8sCrawler**: 专门针对Kubernetes文档的异步爬虫
- **AsyncDocCrawler**: 通用异步文档爬虫框架
- **doc_crawler**: 基础文档爬虫实现
- **crawler_analysis**: 爬虫数据分析工具

### 2. 工具模块 (utils/)
- **MarkdownTreeParser**: 将Markdown文档解析为树状结构，支持智能切分和元数据提取
- **document_schema**: 定义文档数据结构和模式
- **chunker_utils**: 文档块切分工具，保持语义完整性
- **html2md_utils**: HTML到Markdown的转换工具
- **rag_utils**: RAG（检索增强生成）相关实用工具
- **milvus_adapter**: Milvus数据库操作适配器
- **model_factory**: LLM和嵌入模型工厂
- **mcp_manager**: Model Context Protocol管理器

### 3. 工作流模块 (workflow/)
- **build_knowledge_base**: 知识库构建工作流，包括文档解析、向量化和存储

### 4. 检索器模块 (retriever/)
- **MilvusHybridRetriever**: 支持密集向量、稀疏向量和标题向量的混合检索器
- **GraphTraverser**: 图遍历检索器，支持基于文档结构的检索

### 5. Agent智能体模块 (agent/)
- **nodes/**: 包含7个核心节点，实现完整的ReAct工作流
  - **sensory_node**: 感知节点，接收用户输入并注入环境上下文
  - **analysis_node**: 分析节点，理解用户意图、提取实体、评估风险
  - **planning_node**: 规划节点，制定执行计划
  - **retrieval_node**: 检索节点，从知识库检索相关信息
  - **tool_node**: 工具节点，调用MCP工具执行系统操作
  - **regulation_node**: 自我调节节点，评估执行结果并决定下一步
  - **expression_node**: 表达节点，生成最终用户响应
  - **rerank_node**: 重排序节点，使用专用模型对检索结果进行重排序
- **graph**: Agent工作流图定义
- **state**: Agent状态管理
- **schemas**: 数据结构定义
- **prompts**: 动态提示词模板

### 6. 信息桥接模块 (informer/)
- **RuntimeBridge**: 运行时桥接器，连接不同组件和外部服务

### 7. 操作系统MCP服务 (os_mcp/)
- **os_mcp_server**: 基于MCP协议的系统操作服务，提供安全的命令执行和文件操作

### 8. 配置和测试模块
- **config/**: MCP服务配置
- **test/**: 全面的单元测试和集成测试套件

## 系统架构特色

### 1. 混合检索技术
系统采用三路混合检索：
- **密集向量检索**: 基于语义相似性的向量检索
- **稀疏向量检索**: 基于关键词匹配的检索
- **标题向量检索**: 专门针对标题的稀疏向量检索
- **RRF融合**: 使用倒数排名融合算法整合多路检索结果

### 2. ReAct智能Agent
实现完整的ReAct（Reasoning and Acting）模式：
- **Reasoning**: 通过分析和规划节点进行深度推理
- **Acting**: 通过检索和工具调用节点执行具体操作
- **Self-Regulation**: 自我评估和调节机制
- **Adaptive Flow**: 根据评估结果动态调整执行路径

### 3. MCP协议集成
- 支持Model Context Protocol，允许AI模型与外部工具安全交互
- 提供命令执行、文件操作、系统信息查询等功能
- 实现沙箱安全机制，限制操作范围

### 4. 文档结构化处理
- Markdown树状解析，保留文档层次结构
- 智能切分算法，保持语义完整性
- 丰富的元数据提取，支持结构化查询

## 数据库设计

Milvus知识库采用混合检索方案，包含以下字段：

| 字段名 | 字段类型 | 说明 |
|--------|----------|------|
| pk | VarChar | 主键 |
| text | VarChar | 文档文本内容 |
| vector | FloatVector | 文本的稠密向量表示（Qwen生成） |
| summary_vector | FloatVector | 摘要语义向量（Qwen生成） |
| sparse_vector | SparseFloatVector | 文本的稀疏向量表示（BGE-M3生成） |
| title_sparse | SparseFloatVector | 标题的稀疏向量表示（BGE-M3生成） |
| source | VarChar | 源文件路径 |
| title | VarChar | 节点标题 |
| parent_id | VarChar | 父节点ID |
| child_ids | Array[VarChar] | 子节点列表（最大容量256） |
| node_type | VarChar | 节点类型 |
| level | Int64 | 节点层级 |
| token_count | Int64 | token数量 |
| left_sibling | VarChar | 左兄弟节点ID |
| right_sibling | VarChar | 右兄弟节点ID |
| from_split | Bool | 是否来自切分 |
| merged | Bool | 是否合并节点 |
| nav_next_step | VarChar | "接下来"章节内容 |
| nav_see_also | VarChar | "另请参见"章节内容 |
| entry_urls | Array[VarChar] | 超链接入口列表（最大容量200） |
| related_links | JSON | 解析后的关联链路 |
| summary | VarChar | 摘要 |

### 索引配置

| 索引名称 | 索引类型 | 索引参数 |
|----------|----------|----------|
| vector | FLAT/HNSW | 根据配置决定，metric_type: COSINE |
| summary_vector | FLAT/HNSW | 根据配置决定，metric_type: COSINE |
| sparse_vector | SPARSE_INVERTED_INDEX | metric_type: IP, drop_ratio_build: 0.2 |
| title_sparse | SPARSE_INVERTED_INDEX | metric_type: IP, drop_ratio_build: 0.2 |
| pk | 自动 | 主键索引 |
| source | 二级索引 | 用于快速查找源文档 |
| node_type | 二级索引 | 用于按节点类型筛选 |
| level | 二级索引 | 用于按层级筛选 |
| title | 二级索引 | 用于标题搜索 |

## 安装与部署

### 环境要求
- Python 3.8+
- Docker & Docker Compose
- Milvus向量数据库
- PostgreSQL (可选)

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 启动Milvus数据库
```bash
docker-compose -f docker-compose-milvus.yaml up -d
```

### 3. 启动PostgreSQL (如需要)
```bash
docker-compose -f docker-compose-pg.yaml up -d
```

### 4. 爬取文档
```python
from crawler.K8sCrawler import K8sCrawler

# 初始化爬虫
crawler = K8sCrawler(num_workers=5, save_dir="../raw_data")
# 开始爬取
crawler.run()
```

### 5. 构建知识库
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

## 配置文件

项目使用多个配置文件来管理不同组件的配置：

### 1. 环境变量配置 (.env)

位于项目根目录的`.env`文件存储数据库连接等配置信息：

```env
# Milvus数据库配置
MILVUS_HOST=localhost        # Milvus服务器地址
MILVUS_PORT=19530           # Milvus服务端口
MILVUS_USER=root            # Milvus用户名
MILVUS_ROOT_PASSWORD=your_password  # Milvus密码

# PostgreSQL数据库配置（可选）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=kubage_db
```

### 2. MCP服务配置 (config/mcp_config.json)

MCP（Model Context Protocol）服务的配置文件：

```json
{
  "mcpServers": {
    "os-utils": {
      "command": "python",
      "args": ["os_mcp/os_mcp_server.py"],
      "workspace_root": "./workspace"
    }
  }
}
```

配置说明：
- **command**: MCP服务器启动命令
- **args**: 启动参数，指向MCP服务器脚本
- **workspace_root**: 工作空间根目录，限制文件操作范围

### 3. 知识库构建配置

通过`build_knowledge_base`函数参数配置：

```python
build_knowledge_base(
    embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",  # 密集嵌入模型路径
    sparse_model_path="BAAI/bge-m3",                              # 稀疏嵌入模型路径
    markdown_folder_path="../raw_data",                           # 文档源路径
    collection_name="knowledge_base_v2",                          # Milvus集合名称
    max_tokens_per_batch=2048,                                    # 批处理token上限
    min_chunk_size=256,                                           # 最小文档块大小
    core_chunk_size=512,                                          # 核心文档块大小
    max_chunk_size=2048,                                          # 最大文档块大小
    index_type="FLAT",                                            # 向量索引类型
    metric_type="COSINE"                                          # 相似度度量方式
)
```

## 使用示例

### 运行Agent
```python
from agent.graph import build_react_agent
from utils.model_factory import get_chat_model
from retriever.MilvusHybridRetriever import MilvusHybridRetriever

# 初始化组件
llm = get_chat_model()
retriever = MilvusHybridRetriever.from_existing_index(
    collection_name="knowledge_base_v2",
    embedding_field="vector",
    sparse_embedding_field="sparse_vector",
    title_sparse_field="title_sparse"
)

# 构建Agent
app = build_react_agent(llm, retriever)

# 运行查询
result = app.invoke({
    "messages": [("user", "我的Pod一直处于Pending状态，请帮我诊断问题")]
})
```

### 使用检索器
```python
from test.retrieving_test import main

# 运行检索测试
main()
```

## Agent架构详解

### Agent ReAct模式

Agent采用经典的ReAct（Reasoning and Acting）模式，结合了推理（Reasoning）和行动（Acting）两个核心能力：
- **Reasoning**: 通过分析节点和规划节点进行深度思维链推理
- **Acting**: 通过检索、工具调用等节点执行具体操作
- **Self-Regulation**: 通过自我调节节点评估执行结果并决定下一步行动
- **Adaptive Flow**: 支持根据评估结果动态调整执行路径，形成闭环反馈机制

### Agent状态结构 (AgentState)

Agent使用TypedDict定义状态结构，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| messages | Annotated[List[BaseMessage], operator.add] | 消息历史记录，使用operator.add实现追加模式 |
| analysis | ProblemAnalysis \| None | 存储问题分析结果 |
| plan | ExecutionPlan \| None | 存储当前执行计划 |
| retrieved_docs | List[Document] \| None | 检索到的文档片段 |
| retrieval_attempts | int | 检索尝试次数，用于熔断机制 |
| tool_output | str \| None | 工具执行输出结果 |
| evaluation | SelfEvaluation \| None | 自我评估结果 |
| error | str \| None | 错误信息 |
| metadata | Dict[str, Any] \| None | 元数据信息 |

### Agent节点功能

#### 1. Sensory Node（感知节点）
- 接收用户输入，注入环境上下文信息
- 获取系统信息（操作系统、架构、工作空间路径等）

#### 2. Analysis Node（分析节点）
- 深度分析用户输入，提取关键信息
- 进行上下文消歧、实体提取、意图识别和风险评估
- 生成技术摘要和检索Query列表

#### 3. Planning Node（规划节点）
- 基于分析结果制定执行计划
- 支持三种行动类型：RETRIEVE（检索）、TOOL_USE（工具调用）、DIRECT_ANSWER（直接回答）
- 实现检索次数熔断机制，防止无限循环

#### 4. Retrieval Node（检索节点）
- 基于规划节点的搜索查询，从Milvus知识库中检索相关文档
- 对检索结果进行去重处理
- 调用重排序节点对结果进行精排

#### 5. ToolCall Node（工具调用节点）
- 执行MCP注册的工具函数
- 处理命令执行、文件操作等系统级任务

#### 6. Regulation Node（自我调节节点）
- 评估执行结果质量并决定下一步流向
- 根据评估结果决定是否返回分析、规划、检索、工具调用或表达

#### 7. Expression Node（表达节点）
- 生成最终的用户响应
- 根据评估结果决定是否返回反馈信息

### 数据Schema定义

#### 问题分析Schema (ProblemAnalysis)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 思维链推理过程：上下文消歧、意图分析、信息提取、风险判断 |
| entities | List[NamedEntity] | 提取的关键命名实体列表 |
| target_operation | OperationType | 目标操作类型 |
| technical_summary | str | 用户问题的技术摘要，去除口语化表达 |
| risk_level | RiskLevel | 操作风险等级 |
| clarification_question | str \| None | 如果信息不足需要追问的问题 |

#### 执行计划Schema (ExecutionPlan)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 规划的理由，为什么选择这个动作 |
| action | PlanAction | 下一步的具体动作类型 |
| search_queries | List[str] \| None | 检索时使用的查询列表 |
| tool_name | str \| None | 工具调用时的工具名称 |
| tool_args | Dict[str, Any] \| None | 工具调用时的参数 |
| final_answer | str \| None | 直接回答时的内容 |

#### 自我评估Schema (SelfEvaluation)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 评估理由 |
| status | EvaluatedStatus | 当前步骤执行结果的评估状态 |
| next_step | NextStep | 决定回退到哪一步或继续前进 |
| feedback | str | 反馈给下一步骤的改进建议或错误信息 |

## Agent架构

### Agent概述

KubAge新增了基于ReAct模式的智能Agent模块，专门用于理解和分析用户在Kubernetes运维场景中的问题，并提供智能化的解决方案。Agent采用多节点协同工作的方式，通过感知输入、分析问题、规划行动、执行操作、自我调节和表达结果的循环流程来处理复杂的运维任务。

### Agent ReAct模式

Agent采用经典的ReAct（Reasoning and Acting）模式，结合了推理（Reasoning）和行动（Acting）两个核心能力：
- **Reasoning**: 通过分析节点和规划节点进行深度思维链推理
- **Acting**: 通过检索、工具调用等节点执行具体操作
- **Self-Regulation**: 通过自我调节节点评估执行结果并决定下一步行动
- **Adaptive Flow**: 支持根据评估结果动态调整执行路径，形成闭环反馈机制

### Agent状态结构 (AgentState)

Agent使用TypedDict定义状态结构，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| messages | Annotated[List[BaseMessage], operator.add] | 消息历史记录，使用operator.add实现追加模式 |
| analysis | ProblemAnalysis \| None | 存储问题分析结果 |
| plan | ExecutionPlan \| None | 存储当前执行计划 |
| retrieved_docs | List[Document] \| None | 检索到的文档片段 |
| retrieval_attempts | int | 检索尝试次数，用于熔断机制 |
| tool_output | str \| None | 工具执行输出结果 |
| evaluation | SelfEvaluation \| None | 自我评估结果 |
| error | str \| None | 错误信息 |
| metadata | Dict[str, Any] \| None | 元数据信息 |

### Agent节点 (Nodes)

Agent包含以下几种节点类型，每个节点负责不同的处理任务：

#### 1. Sensory Node（感知节点）
- **功能**: 接收用户输入，注入环境上下文信息
- **输入**: 用户消息
- **输出**: 包含环境信息的消息状态
- **处理流程**:
  1. 检查是否已注入环境上下文
  2. 获取系统信息（操作系统、架构、工作空间路径等）
  3. 将环境信息作为SystemMessage注入到消息历史中

#### 2. Analysis Node（分析节点）
- **功能**: 对用户输入进行深度分析，提取关键信息
- **输入**: 用户消息历史和当前输入
- **输出**: 结构化的ProblemAnalysis对象
- **处理流程**:
  1. 使用JsonOutputParser解析ProblemAnalysis模型
  2. 通过思维链推理进行上下文消歧、实体提取、意图识别和风险评估
  3. 生成技术摘要和检索Query列表
  4. 输出包含完整推理过程的结构化分析结果

#### 3. Planning Node（规划节点）
- **功能**: 基于分析结果制定执行计划
- **输入**: 问题分析、文档知识、上轮计划、评估反馈
- **输出**: ExecutionPlan对象
- **处理流程**:
  1. 根据状态动态生成系统指令（强制检索模式、工具修复模式、标准模式）
  2. 使用动态提示词构造生成规划决策
  3. 支持三种行动类型：RETRIEVE（检索）、TOOL_USE（工具调用）、DIRECT_ANSWER（直接回答）
  4. 实现检索次数熔断机制，防止无限循环

#### 4. Retrieval Node（检索节点）
- **功能**: 基于规划节点的搜索查询，从Milvus知识库中检索相关文档
- **输入**: 规划结果中的search_queries
- **输出**: 检索到的相关文档片段列表
- **处理流程**:
  1. 获取上一节点的规划结果
  2. 遍历所有搜索查询进行检索
  3. 对检索结果进行去重处理
  4. 调用重排序节点对结果进行精排
  5. 返回最终的文档片段列表

#### 5. ToolCall Node（工具调用节点）
- **功能**: 执行MCP注册的工具函数
- **输入**: 规划结果中的工具名称和参数
- **输出**: 工具执行结果
- **处理流程**:
  1. 从MCP工具管理器获取对应工具函数
  2. 构造AI消息和工具消息记录交互过程
  3. 异步执行工具调用
  4. 解析和格式化工具输出结果
  5. 将结果记录到消息历史中

#### 6. Regulation Node（自我调节节点）
- **功能**: 评估执行结果质量并决定下一步流向
- **输入**: 分析结果、执行计划、执行结果
- **输出**: SelfEvaluation对象
- **处理流程**:
  1. 根据当前动作类型（检索或工具调用）应用不同的评估标准
  2. 评估结果质量（通过、需改进、失败）
  3. 生成反馈信息和改进建议
  4. 决定下一步行动（返回分析、规划、检索、工具调用或表达）

#### 7. Expression Node（表达节点）
- **功能**: 生成最终的用户响应
- **输入**: 执行计划和评估结果
- **输出**: 最终响应消息
- **处理流程**:
  1. 检查是否存在直接回答内容
  2. 根据评估结果决定是否返回反馈信息
  3. 生成符合用户期望的最终响应

### 数据Schema定义

#### 问题分析Schema (ProblemAnalysis)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 思维链推理过程：上下文消歧、意图分析、信息提取、风险判断 |
| entities | List[NamedEntity] | 提取的关键命名实体列表 |
| target_operation | OperationType | 目标操作类型 |
| technical_summary | str | 用户问题的技术摘要，去除口语化表达 |
| risk_level | RiskLevel | 操作风险等级 |
| clarification_question | str \| None | 如果信息不足需要追问的问题 |

#### 执行计划Schema (ExecutionPlan)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 规划的理由，为什么选择这个动作 |
| action | PlanAction | 下一步的具体动作类型 |
| search_queries | List[str] \| None | 检索时使用的查询列表 |
| tool_name | str \| None | 工具调用时的工具名称 |
| tool_args | Dict[str, Any] \| None | 工具调用时的参数 |
| final_answer | str \| None | 直接回答时的内容 |

#### 自我评估Schema (SelfEvaluation)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| reasoning | str | 评估理由 |
| status | EvaluatedStatus | 当前步骤执行结果的评估状态 |
| next_step | NextStep | 决定回退到哪一步或继续前进 |
| feedback | str | 反馈给下一步骤的改进建议或错误信息 |

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
| INSTALL | 安装 |
| PROXY | 代理 |
| OTHER | 其他操作 |

### 风险等级 (RiskLevel)

| 枚举值 | 说明 |
|--------|------|
| LOW | 信息查询，只读操作 |
| MEDIUM | 配置检查、非破坏性调试 |
| HIGH | 配置变更、重启 |
| CRITICAL | 删除资源、危险操作 |

### 规划行动类型 (PlanAction)

| 枚举值 | 说明 |
|--------|------|
| RETRIEVE | 查询知识库 |
| TOOL_USE | 调用工具 |
| DIRECT_ANSWER | 直接回答/任务结束 |

### 评估状态 (EvaluatedStatus)

| 枚举值 | 说明 |
|--------|------|
| PASS | 执行成功 |
| FAIL | 执行失败 |
| NEEDS_REFINEMENT | 结果不理想，需要优化 |

### 下一步行动 (NextStep)

| 枚举值 | 说明 |
|--------|------|
| TO_ANALYSIS | 重新分析（意图理解有误） |
| TO_PLANNING | 重新规划（更换工具或检索词） |
| TO_RETRIEVAL | 重新检索 |
| TO_TOOL | 重新调用工具 |
| TO_EXPRESSION | 回答用户 |

### 动态提示词构造

Agent实现了高级的动态提示词构造机制：

#### 1. 分层提示词设计
- **基础系统提示**: 定义Agent角色和核心原则
- **动态系统指令**: 根据当前状态切换不同模式（强制检索、工具修复、标准模式）
- **用户提示模板**: 包含问题分析、文档知识、历史计划和评估反馈

#### 2. 模式切换机制
- **强制检索模式**: 适用于高风险操作或知识密集型任务
- **强制工具修复模式**: 适用于工具调用失败需要参数修正
- **标准模式**: 自由规划阶段，可根据信息充足度选择行动

#### 3. 上下文感知
- 根据检索尝试次数实施熔断机制
- 基于评估反馈调整策略
- 考虑历史消息中的工具输出结果

### 图定义和流程控制

#### 工作流图结构
```
START -> Sensory -> Analysis -> [Conditional: Clarification?]
                                -> Expression (if clarification needed)
                                -> Planning -> [Conditional: Action Type]
                                    -> Retrieval -> Self-Regulation -> [Conditional: Next Step]
                                    -> ToolCall -> Self-Regulation -> [Conditional: Next Step]
                                    -> Expression -> END
```

#### 条件分支逻辑
1. **分析分支**: 根据是否需要澄清问题决定是否直接表达
2. **规划分支**: 根据规划动作类型决定流向检索或工具调用
3. **调节分支**: 根据评估结果决定下一步行动方向

#### 循环控制机制
- **检索循环**: 评估失败时返回规划节点重新检索
- **工具循环**: 工具失败时返回规划节点修复参数
- **分析循环**: 意图理解错误时返回分析节点重新理解

### Agent测试验证

Agent的功能通过全面的测试用例进行验证：

#### 节点独立测试
- **分析节点**: 验证实体提取、意图识别、风险评估的准确性
- **规划节点**: 测试动态模式切换和检索Query生成
- **检索节点**: 验证文档去重和重排序效果
- **工具节点**: 测试MCP工具调用和错误处理
- **调节节点**: 验证评估逻辑和路径决策

#### 端到端工作流测试
- **简单问答**: 知识库查询场景
- **复杂运维**: 故障诊断和修复流程
- **工具调用**: 命令执行和结果处理
- **错误恢复**: 异常情况下的流程恢复

#### 具体测试场景包括：
- **故障诊断场景**: Pod启动失败、服务不可达等问题
- **高风险操作识别**: 删除资源、配置变更等需要确认的操作
- **知识问答**: 纯理论问题的处理
- **工具调用**: kubectl命令执行、文件操作等
- **多轮对话**: 需要澄清和迭代的问题
- **错误处理**: 工具失败、检索无结果等情况

### Agent工作流示例

```python
from langgraph.graph import StateGraph, START, END
from agent.graph import build_react_agent
from agent.state import AgentState
from utils.llm_factory import get_chat_model
from retriever.MilvusHybridRetriever import MilvusHybridRetriever
from agent.nodes.rerank_node import RerankNode

# 构建ReAct Agent
llm = get_chat_model()
retriever = MilvusHybridRetriever.from_existing_index(
    collection_name="knowledge_base_v2",
    embedding_field="vector",
    sparse_embedding_field="sparse_vector",
    title_sparse_field="title_sparse"
)
reranker = RerankNode(model_path="../models/Qwen/Qwen3-Reranker-0.6B")

# 构建ReAct工作流
app = build_react_agent(
    llm=llm,
    retriever=retriever,
    reranker=reranker,
    tool_descriptions="可用工具列表描述..."
)

# 运行工作流
result = app.invoke({
    "messages": [HumanMessage(content="帮我排查nginx部署的连接问题")]
})
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