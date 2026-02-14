import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from utils import MarkdownTreeParser, get_dense_embed_model
from utils.chunker_utils import visualize_document_tree

RAW_DATA_DIR = "../raw_data"

qwen_path = "../models/Qwen/Qwen3-Embedding-0.6B"

# 1. 初始化嵌入模型
embeddings = get_dense_embed_model(qwen_path)

# 2. 初始化 MarkdownTreeParser
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
parser = MarkdownTreeParser(
    embeddings=embeddings,
    tokenizer=tokenizer,
    min_chunk_size=256,
    core_chunk_size=512,
    max_chunk_size=2048
)

# 3. 解析所有 Markdown 文件
docs = []
for file in ["../raw_data/文档_参考_Kubernetes API_工作负载资源_Pod.md"]:
    print(f"正在处理：{file}, 已完成：{len(docs)}")
    if file.endswith(".md"):
        file_path = os.path.join(RAW_DATA_DIR, file)
        docs.extend(parser.parse_markdown_to_tree(Path(file_path)))
        torch.cuda.empty_cache()

visualize_document_tree(docs, show_siblings=True)