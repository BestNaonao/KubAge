from typing import List

from langchain_core.documents import Document

from utils import NodeType

RETRIEVAL_PROMPT = """
5. **检索策略生成**: 将用户的自然语言转化为专业的 K8s 术语，以生成高质量的检索 Query。
   - 你将要检索的知识库是《Kubernetes 官方中文文档》。
   - 文档中没有用户的具体实体名称。
   - **必须**将用户的具体问题抽象为通用的 Kubernetes 概念或错误类型。
   - **禁止**在 Query 中包含具体的实体名称。
   - **语言要求**: 生成的 Query 必须使用 **中文** 来描述问题逻辑，但必须 **保留英文** 的 Kubernetes 专有名词。
   - **混合模式**: 最佳的 Query 结构是 “英文术语 + 中文描述”。

# 检索词生成示例 (Few-Shot Examples)
- 用户输入: "我的 redis 总是起不来，状态是 CrashLoopBackOff"
  ❌ 错误 (含实体): ["redis 启动失败", "排查 CrashLoopBackOff 状态的 redis"]
  ❌ 错误 (纯英文): ["Troubleshoot CrashLoopBackOff", "Pod startup failure"]
  ❌ 错误 (纯中文): ["容器崩溃循环排查", "应用启动失败"]
  ✅ 正确 (混合): ["CrashLoopBackOff 排查思路", "Pod 启动失败原因", "容器反复重启 debug"]

- 用户输入: "Service 连不上 Pod"
  ✅ 正确 (混合): ["Service 连接 Pod 超时", "Service debug 步骤", "Pod 网络不通排查"]
"""

# 普通文档模板
DOC_PROMPT_TEMPLATE = """【文档 {index}】
标题: {title}
内容: 
{content}
"""

# 动态事件专用模板
DYNAMIC_EVENT_TEMPLATE = """【故障现场】
事件: {title}
详细日志: {content}
关联指导: {related_guide}
"""

def format_docs(docs: List[Document]) -> str:
    """
    将文档列表格式化为字符串，供 LLM 审查。
    To Do: 限制总字符数防止 Context 溢出。
    """
    if not docs:
        return "No documents retrieved."

    formatted_entries = []
    # 1. 排序：强制将 dynamic_event 排在最前面，引起 LLM 注意
    sorted_docs = sorted(docs, key=lambda d: 0 if d.metadata.get("node_type") == NodeType.EVENT else 1)

    for i, doc in enumerate(sorted_docs):
        content = doc.page_content.strip()
        title = doc.metadata.get('title', 'Unknown')

        if doc.metadata["node_type"] == NodeType.EVENT:
            # 提取关联信息，告诉 LLM 这个报错对应哪个手册
            related_guide = "None"
            for link in doc.metadata.get("related_links", []):
                if link.get("type") == "static_anchor":
                    related_guide = link.get("text")
                    break

            entry = DYNAMIC_EVENT_TEMPLATE.format(title=title, content=content, related_guide=related_guide)
        else:
            entry = DOC_PROMPT_TEMPLATE.format(index=i + 1, title=title, content=content)

        formatted_entries.append(entry)

    return "\n\n".join(formatted_entries)

def format_reflections(reflections: list[str]) -> str:
    if reflections:
        return "\n".join([f"  {i + 1}. {r}" for i, r in enumerate(reflections)])
    else:
        return "None"
