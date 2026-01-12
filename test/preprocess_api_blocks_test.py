import unittest
import re

from utils import MarkdownTreeParser


class TestApiPreprocessor(unittest.TestCase):

    @staticmethod
    def _preprocess_api_blocks(content: str) -> str:
        """
        待测试的核心逻辑函数。
        直接从 MarkdownTreeParser 中提取出来的逻辑。
        """
        return MarkdownTreeParser.preprocess_api_blocks(content)

    def test_basic_api_block(self):
        """测试用例 1: 标准的三要素 API 块"""
        raw_md = """
### 读取 Pod
#### HTTP 请求
GET /api/v1/pods
#### 参数
name: string
#### 响应
200 OK
"""
        expected_md = """
### 读取 Pod
**HTTP 请求**
GET /api/v1/pods
**参数**
name: string
**响应**
200 OK
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 标准 API 块测试通过")

    def test_sinking_logic(self):
        """测试用例 2: 下沉逻辑 (Parent Level Shift)
        测试 ## 操作 -> ### API 这种结构，能否正确识别 ### 为真正的父标题
        """
        raw_md = """
## 操作
这里是一些描述。
### `get` 读取 Pod
#### HTTP 请求
GET /...
#### 响应
200 OK
"""
        # 预期：
        # ## 操作 保持不变
        # ### get ... 保持不变
        # #### HTTP 请求 变成 **HTTP 请求**
        # #### 响应 变成 **响应**

        expected_md = """
## 操作
这里是一些描述。
### `get` 读取 Pod
**HTTP 请求**
GET /...
**响应**
200 OK
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 下沉逻辑 (父标题自动修正) 测试通过")

    def test_incomplete_block_ignored(self):
        """测试用例 3: 不完整的块不应被修改
        防止误伤普通的文档结构（例如某处只有一个 '#### 参数' 章节）
        """
        raw_md = """
### 普通章节
#### 参数
这里只是介绍了一些参数，但没有HTTP请求和响应。
#### 其他内容
...
"""
        # 预期：没有任何变化，因为没有集齐2个以上的关键词
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), raw_md.strip())
        print("✅ 非 API 块误伤保护测试通过")

    def test_multiple_blocks(self):
        """测试用例 4: 同一文件多个 API 块"""
        raw_md = """
### API A
#### HTTP 请求
A
#### 响应
A

### API B
#### HTTP 请求
B
#### 参数
B
#### 响应
B
"""
        expected_md = """
### API A
**HTTP 请求**
A
**响应**
A

### API B
**HTTP 请求**
B
**参数**
B
**响应**
B
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 多块处理测试通过")

    def test_english_headers(self):
        """测试用例 5: 英文标题支持"""
        raw_md = """
### Read Pod
#### HTTP Request
GET /...
#### Response
200 OK
"""
        expected_md = """
### Read Pod
**HTTP Request**
GET /...
**Response**
200 OK
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 英文标题支持测试通过")

    def test_mishierarchicaled_headers(self):
        """测试用例 6: API 块的标题等级错误"""
        raw_md = """
### API A
#### HTTP 请求
A
##### 响应
A

### API B
##### HTTP 请求
B
###### 参数
B
#### 响应
B
"""
        expected_md = """
### API A
**HTTP 请求**
A
**响应**
A

### API B
**HTTP 请求**
B
**参数**
B
**响应**
B
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 标题等级错误修正通过")

    def test_weird_keyword_variation(self):
        """测试用例 7: 非标准的 'HTTP 参数' 标题"""
        raw_md = """
### `list` 列出对象
#### HTTP 参数
GET /api/v1/...
#### 参数
name: string
#### 响应
200 OK
"""
        # 预期：HTTP 参数、参数、响应 都被降级
        # 注意：虽然 "HTTP 参数" 和 "参数" 看起来重复，但只要逻辑能跑通即可
        expected_md = """
### `list` 列出对象
**HTTP 参数**
GET /api/v1/...
**参数**
name: string
**响应**
200 OK
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 变种关键词 'HTTP 参数' 测试通过")

    def test_level_mismatch_exemption(self):
        """测试用例 8: 层级错乱豁免 (Keyword Exemption)
        测试 ### 参数 出现在 ### delete 下面（同级），不应截断
        """
        raw_md = """
### `delete` 删除策略
#### HTTP 请求
DELETE /...
### 参数
name: string
#### 响应
200 OK
"""
        # 预期：即使 '### 参数' 是三级标题（和父级同级），也应该被识别并降级
        expected_md = """
### `delete` 删除策略
**HTTP 请求**
DELETE /...
**参数**
name: string
**响应**
200 OK
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 层级错乱豁免 (### 参数) 测试通过")

    def test_keyword_normalization(self):
        """测试用例 9: 关键词归一化
        防止 'HTTP 请求' 和 'HTTP 参数' 同时出现时被误判为两个不同要素
        (虽然这种情况很少见，但逻辑上要保证健壮性)
        """
        raw_md = """
### 怪异的块
#### HTTP 请求
GET /...
#### HTTP 参数
...
"""
        # 预期：只收集到了 "HTTP" 这一种类型的要素，数量为1 (<2)，不应降级
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), raw_md.strip())
        print("✅ 关键词归一化 (防止重复计算 HTTP) 测试通过")

    def test_standard_english(self):
        """测试用例 10: 标准英文结构"""
        raw_md = """
### Read Pod
#### HTTP Request
GET /...
#### Parameters
name: string
#### Response
200 OK
"""
        expected_md = """
### Read Pod
**HTTP Request**
GET /...
**Parameters**
name: string
**Response**
200 OK
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 英文标准结构测试通过")

    def test_complex_document(self):
        """测试用例 11: 复杂混合文档"""
        raw_md = """
# API 参考
## Workloads
### Pod
#### HTTP
GET /...
#### 响应
200

### Service
#### HTTP 参数
GET /...
### 参数
name: string
#### 响应
200
"""
        expected_md = """
# API 参考
## Workloads
### Pod
**HTTP**
GET /...
**响应**
200

### Service
**HTTP 参数**
GET /...
**参数**
name: string
**响应**
200
"""
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), expected_md.strip())
        print("✅ 复杂多块混合文档测试通过")

    def test_false_positive_prevention(self):
        """测试用例 12: 防止模糊匹配误伤
        'HTTP 协议' 不在白名单中，不应被降级
        """
        raw_md = """
### 介绍
#### HTTP 协议历史
这是一个介绍章节。
#### 响应式设计
这也是介绍。
"""
        # 预期：不发生任何改变
        result = self._preprocess_api_blocks(raw_md.strip())
        self.assertEqual(result.strip(), raw_md.strip())
        print("✅ 白名单防误伤测试通过")


if __name__ == '__main__':
    unittest.main(verbosity=2)