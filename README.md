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
├── agent/                   # Agent模块
│   ├── nodes/               # Agent节点
│   │   ├── analysis_node.py    # 问题分析节点
│   │   ├── retrieval_node.py   # 文档检索节点
│   │   └── rerank_node.py      # 文档重排序节点
│   ├── graph.py             # Agent工作流图
│   ├── state.py             # Agent状态定义
│   ├── schemas.py           # 数据结构定义
│   └── prompts.py           # Prompt模板
├── config/                  # 配置模块
│   └── mcp_config.json      # MCP服务配置
├── os_mcp/                  # 操作系统MCP服务
│   └── os_mcp_server.py     # MCP服务器实现
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

### 5. Agent问题分析
- **工具**: `agent/nodes/analysis_node.py`
- **功能**:
  - 理解用户的Kubernetes运维问题
  - 提取关键实体（Pod名称、Namespace等）
  - 识别操作类型和风险等级
  - 生成优化的检索Query

### 6. 混合检索
- **工具**: `retriever/MilvusHybridRetriever.py` + `agent/nodes/retrieval_node.py`
- **功能**:
  - 基于密集向量进行语义检索
  - 基于稀疏向量进行关键词匹配
  - 基于标题稀疏向量进行标题匹配
  - 使用RRF算法融合多路检索结果

### 7. 文档重排序
- **工具**: `agent/nodes/rerank_node.py`
- **功能**:
  - 使用Qwen3-Reranker模型对检索结果重新排序
  - 根据操作类型生成动态排序指令
  - 提高最终结果的相关性和准确性

### 8. 命令执行（通过os_mcp）
- **工具**: `os_mcp/os_mcp_server.py`
- **功能**:
  - 提供MCP（Model Context Protocol）服务接口
  - 支持在安全沙箱环境中执行系统命令
  - 支持文件读写操作
  - 支持环境变量管理
  - 为Agent提供与操作系统交互的能力

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

## OS MCP服务 (os_mcp)

### 概述

`os_mcp`是基于MCP（Model Context Protocol）协议实现的操作系统工具服务，为Agent提供与操作系统安全交互的能力。该服务运行在受限的沙箱环境中，确保命令执行的安全性。

### 服务架构

- **位置**: `os_mcp/os_mcp_server.py`
- **协议**: 基于FastMCP实现的MCP服务器
- **沙箱**: 所有文件操作限制在`./workspace`目录内
- **异步**: 使用asyncio实现异步命令执行

### 核心功能

#### 1. 命令执行 (execute_command)
```python
execute_command(command: str, timeout: int = 60)
```
- 异步执行系统shell命令
- 支持超时控制（默认60秒）
- 自动处理stdout/stderr输出
- 返回退出码和完整输出信息
- **安全限制**: 不支持管道(|)和重定向(>)操作

#### 2. 文件读取 (read_file)
```python
read_file(path: str)
```
- 读取工作空间内的文件内容
- 自动检测文件编码（UTF-8、GBK、Latin-1）
- 路径必须在允许的工作空间范围内

#### 3. 文件写入 (write_file)
```python
write_file(path: str, content: str)
```
- 写入内容到工作空间文件
- 自动创建不存在的目录
- 覆盖已存在的文件

#### 4. 系统信息获取 (get_system_info)
```python
get_system_info()
```
- 返回操作系统类型、版本、架构等信息
- 提供工作空间绝对路径
- 用于环境兼容性检查

#### 5. 环境变量管理 (append_environment_variable)
```python
append_environment_variable(key: str, value: str)
```
- 将环境变量追加到`.env`文件
- 持久化保存配置信息

### 安全特性

1. **路径验证**: 所有文件操作都经过严格的路径验证，禁止访问工作空间外的文件
2. **命令沙箱**: 使用shlex解析命令，防止命令注入攻击
3. **超时保护**: 所有命令执行都有超时限制，防止进程挂起
4. **错误隔离': 单个操作失败不会影响整个服务

### 使用场景

- **Kubernetes运维**: 执行kubectl命令查看集群状态
- **配置管理**: 读写YAML/JSON配置文件
- **日志分析**: 读取和分析容器日志
- **脚本执行**: 运行诊断和修复脚本

## 数据处理流程

### 知识库构建流程

1. **爬虫阶段**: 从Kubernetes官网爬取HTML文档
2. **格式转换**: 将HTML转换为Markdown格式
3. **文档解析**: 解析Markdown为树状结构
4. **内容切分**: 按语义和大小切分文档块
5. **密集向量化**: 使用Qwen模型生成密集向量表示
6. **稀疏向量化**: 使用BGE-M3模型生成稀疏向量表示
7. **存储**: 将向量和元数据存储到Milvus
8. **索引**: 为向量和元数据建立索引

### 智能问答流程

1. **问题输入**: 用户提出Kubernetes运维相关问题
2. **Agent分析**: 分析问题意图、提取实体、评估风险、生成检索Query
3. **混合检索**: 使用密集向量、稀疏向量和标题向量进行三路检索
4. **结果融合**: 使用RRF算法融合多路检索结果
5. **文档重排**: 使用Reranker模型对结果重新排序
6. **答案生成**: 基于检索到的知识生成回答
7. **命令执行**: 通过os_mcp服务执行必要的kubectl命令或系统操作
8. **结果反馈**: 将执行结果返回给用户

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