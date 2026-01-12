import os
import re

from transformers import AutoTokenizer


def extract_api_blocks(directory_path):
    """
    遍历指定目录下的 md 文件，提取符合特定结构的 API 文本块。
    支持同一文件提取多个块，且自动识别不同的父标题等级。
    """
    # 定义必需的子标题关键词 (根据需要可以调整，这里做去格式化后的精确匹配)
    REQUIRED_SUBHEADERS = {"HTTP 请求", "参数", "响应"}

    # 用于存储所有提取出的块
    all_extracted_blocks = []

    # 遍历目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    blocks = parse_markdown_file(file_path, REQUIRED_SUBHEADERS)
                    if blocks:
                        all_extracted_blocks.extend(blocks)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    return all_extracted_blocks


def parse_markdown_file(file_path, required_subheaders):
    """
    解析单个 Markdown 文件，提取目标块。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    extracted = []

    # 状态变量
    current_parent_level = 0
    current_block_lines = []
    found_subheaders = set()

    # 正则：匹配 Markdown 标题
    header_pattern = re.compile(r'^(#+)\s+(.*)')

    for line in lines:
        match = header_pattern.match(line)

        if match:
            level = len(match.group(1))
            raw_text = match.group(2).strip()
            # 去除可能的加粗、反引号等 Markdown 符号，用于关键词比对
            clean_text = raw_text.replace('**', '').replace('`', '').strip()

            # --- 逻辑判断核心 ---

            # 1. 检查边界：如果遇到了同级或更高级（数字更小）的标题
            #    这意味着当前正在累积的块必须结束（无论是否成功）
            if current_parent_level > 0 and level <= current_parent_level:
                # 结算上一个块
                if required_subheaders.issubset(found_subheaders):
                    extracted.append({
                        "source": os.path.basename(file_path),  # 只保留文件名，或者用 file_path 保留全路径
                        "content": "".join(current_block_lines)
                    })

                # 重置状态
                current_parent_level = 0
                current_block_lines = []
                found_subheaders = set()

            # 2. 判断当前行是否是关键词子标题（HTTP/参数/响应）
            is_keyword_header = clean_text in required_subheaders

            # 3. 处理新标题进入逻辑
            if current_parent_level == 0:
                # 如果当前没有在录制块，且标题等级 >= 2，则将其视为潜在的父标题开始录制
                if level >= 2:
                    current_parent_level = level
                    current_block_lines = [line]
            else:
                # 如果当前正在录制 (current_parent_level > 0) 且 level > current_parent_level (是子标题)

                if is_keyword_header:
                    # 如果是关键词子标题，记录它，并继续录入
                    found_subheaders.add(clean_text)
                    current_block_lines.append(line)
                else:
                    # 关键点：如果是一个普通的子标题（不是关键词），而且我们还没收集到任何关键词
                    # 这说明之前的“父标题”可能只是一个大的容器（比如 ## 操作），
                    # 而当前的这个标题（比如 ### get xxx）才是真正的 API 块开头。
                    # 我们执行“下沉”操作：丢弃之前的行，将当前行作为新的父标题。
                    if not found_subheaders:
                        current_parent_level = level
                        current_block_lines = [line]  # 重新开始
                    else:
                        # 如果已经收集到了关键词，中间又出现了其他非关键词子标题，
                        # 通常视为内容的一部分，照常追加
                        current_block_lines.append(line)

        else:
            # --- 非标题行 ---
            if current_parent_level > 0:
                current_block_lines.append(line)

    # --- 循环结束后，处理文件末尾的最后一个块 ---
    if current_parent_level > 0 and required_subheaders.issubset(found_subheaders):
        extracted.append({
            "source": os.path.basename(file_path),
            "content": "".join(current_block_lines)
        })

    return extracted


if __name__ == "__main__":
    # 参数
    qwen_path = "../models/Qwen/Qwen3-Embedding-0.6B"
    folder = "../raw_data"

    # 2. 初始化 MarkdownTreeParser
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    if not os.path.exists(folder):
        print(f"错误: 文件夹 '{folder}' 不存在。")
    else:
        results = extract_api_blocks(folder)
        results.sort(key=lambda x: len(tokenizer.tokenize(x['content'])), reverse=True)

        print(len(results))

        for result in results:
            print(result['source'])
            print(result['content'])
            print(f"token_count: {len(tokenizer.tokenize(result['content']))}")
            print("=========")

