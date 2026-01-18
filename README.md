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