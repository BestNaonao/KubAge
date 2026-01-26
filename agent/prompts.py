from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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

