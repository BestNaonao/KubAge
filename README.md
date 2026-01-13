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
│   └── html2md_utils.py      # HTML转Markdown工具
├── test/                    # 测试模块
│   └── build_kb_test.py     # 知识库构建测试
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
- **工具**: `test/build_kb_test.py`
- **功能**:
  - 使用嵌入模型生成向量表示
  - 将文档数据存储到Milvus向量数据库
  - 支持批量处理和错误恢复

## 数据库结构

Milvus知识库包含16个字段，具体如下：

| 字段名 | 字段类型 | 说明 |
|--------|----------|------|
| text | VarChar | 文档文本内容 |
| pk | Int64 | 主键 |
| vector | FloatVector | 文档向量表示 |
| source | VarChar | 源文件路径 |
| parent_id | VarChar | 父节点ID |
| child_ids | VarChar | 子节点ID列表（JSON格式） |
| node_type | VarChar | 节点类型（ROOT、SECTION、CONTAINER、LEAF） |
| level | Int64 | 节点层级 |
| title | VarChar | 节点标题 |
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
| pk | 自动 | 主键索引 |
| source | 二级索引 | 用于快速查找源文档 |
| node_type | 二级索引 | 用于按节点类型筛选 |
| level | 二级索引 | 用于按层级筛选 |
| title | 二级索引 | 用于标题搜索 |

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
from test.build_kb_test import build_knowledge_base

# 使用默认参数构建知识库
build_knowledge_base(
    embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",
    markdown_folder_path="../raw_data",
    collection_name="knowledge_base_v1",
    max_tokens_per_batch=2048,
    milvus_host="localhost",
    milvus_port=19530
)
```

### build_knowledge_base函数参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| embedding_model_path | "../models/Qwen/Qwen3-Embedding-0.6B" | 嵌入模型路径 |
| markdown_folder_path | "../raw_data" | Markdown文件夹路径 |
| collection_name | "knowledge_base_v1" | Milvus集合名 |
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
5. **向量化**: 使用嵌入模型生成向量表示
6. **存储**: 将向量和元数据存储到Milvus
7. **索引**: 为向量和元数据建立索引

## 依赖包

主要依赖包包括：
- `langchain-huggingface`: HuggingFace嵌入模型集成
- `langchain-milvus`: Milvus向量数据库集成
- `transformers`: 模型和分词器
- `torch`: PyTorch深度学习框架
- `pymilvus`: Milvus Python SDK
- `beautifulsoup4`: HTML解析
- `python-dotenv`: 环境变量管理