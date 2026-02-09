import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from utils.html2md_utils import convert_to_markdown, get_span_path

HEADERS: Dict[str, str] = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Connection": "keep-alive"
}
NO_REF_URL: str = "https://kubernetes.io"
BASE_URL: str = "https://kubernetes.io/zh-cn/docs/"
START_URL: str = BASE_URL + "home/"
ELEMENT_NAME_LIST: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'code']
EXCLUDE_HREF = ["/contribute", "/blog", "/training", "/careers", "/partners", "/community", "/test",
                "/feature-gates-removed", "/reference/issues-security"]
REMOVE_TEXT: str = "此页是否对你有帮助"


class K8sCrawler(ABC):
    def __init__(self, num_workers: int, save_dir: str):
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.found: Set[str] = set()
        self.errored: List[str] = []  # 只追加
        self.filename_counter = {}
        self.headers: Dict[str, str] = HEADERS
        self.start_url = START_URL

        # 日志配置
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("crawler_async.log", encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def initialize(self):
        self.found.clear()
        self.errored.clear()
        self.found.add(START_URL)

    @staticmethod
    def extract_new_urls(soup, url) -> List[str]:
        """提取页面中的新URL并返回列表"""
        new_urls = []
        parent_elements = soup.select("ul.ul-1")
        for parent_element in parent_elements:
            for link in parent_element.find_all('a', href=True):
                href = link.get('href')
                full_url = urljoin(BASE_URL, link['href'])
                if any(prefix in href for prefix in EXCLUDE_HREF) or not full_url.startswith(BASE_URL):
                    continue
                if '#' in full_url or full_url == url:
                    continue
                new_urls.append(full_url)
        return new_urls

    def get_unique_filename(self, base_name: str) -> str:
        """生成唯一的文件名，避免冲突"""
        # 如果文件名已存在，添加计数器后缀
        if base_name in self.filename_counter:
            self.filename_counter[base_name] += 1
            return f"{base_name}_{self.filename_counter[base_name]}"
        else:
            self.filename_counter[base_name] = 1
            return base_name

    def parse_html(self, html: str, url: str) -> Tuple[List[str], str, str]:
        soup = BeautifulSoup(html, 'lxml')
        new_urls = self.extract_new_urls(soup, url)
        span_name = get_span_path(soup, url, NO_REF_URL)
        markdown_content = convert_to_markdown(soup, url)
        return new_urls, span_name, markdown_content

    def save(self, markdown_content: str, url: str, doc_name: str):
        """提取网页中的内容，包括标题和正文，最后保存"""
        if not markdown_content:
            self.logger.warning(f"⚠️ 未找到正文内容: {url}")
            return

        save_path = os.path.join(self.save_dir, f"{doc_name}.md")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        self.logger.info(f"已保存: {save_path} (字符数: {len(markdown_content)})")

    def show_progress(self, done: int, total: int, active: int):
        if total > 0:
            progress = done / total * 100
            print(
                f"\r📊 进度: {done}/{total} ({progress:.1f}%) | 队列: {total - done} | "
                f"活跃Worker: {active} / {self.num_workers} | 错误: {len(self.errored)}", end='', flush=True
            )

    def quit_print(self):
        print("\n" + "=" * 50)
        self.logger.info(f"爬取完成! 总共访问: {len(self.found)} 页面, 错误: {len(self.errored)}")

        if self.errored:
            self.logger.info("----- 以下URL获取失败:")
            for url in self.errored:
                self.logger.info(f"--- {url}")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def insert_title_at_regex_position(self, file_name: str, title: str, regex_pattern: str):
        """
        在文件中匹配正则表达式的位置插入标题

        参数:
        file_path (str): 文件路径
        title (str): 要插入的标题（带格式，如"###### CEL 表达式规则"）
        regex_pattern (str): 用于搜索位置的正则表达式
        """
        try:
            # 读取文件内容
            file_path = os.path.join(self.save_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # 使用正则表达式搜索匹配位置
            match = re.search(regex_pattern, content, re.MULTILINE)

            if match:
                # 获取匹配位置
                start_pos = match.start()

                # 在匹配位置前插入标题（带换行符）
                new_content = content[:start_pos] + f"\n{title}\n" + content[start_pos:]

                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)

                print(f"成功在文件中插入标题: {title}")
                print(f"插入位置: 第{content[:start_pos].count('\n') + 1}行附近")
                return True
            else:
                print("未找到匹配的位置")
                return False

        except FileNotFoundError:
            print(f"文件不存在: {file_name}")
            return False
        except Exception as e:
            print(f"处理文件时发生错误: {e}")
            return False

    def replace_content_by_regex(self, file_name: str, regex_pattern: str, substitute: str):
        """
        在文件中替换符合正则表达式的内容

        参数:
        file_name (str): 文件名（不含路径）
        regex_pattern (str): 用于匹配内容的正则表达式
        substitute (str): 替换后的字符串内容
        """
        try:
            # 构建完整文件路径
            file_path = os.path.join(self.save_dir, file_name)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # 检查是否有匹配内容
            if not re.search(regex_pattern, content, re.MULTILINE):
                print(f"未找到匹配内容: {regex_pattern}")
                return False

            # 执行替换（替换所有匹配项）
            new_content, count = re.subn(regex_pattern, substitute, content, flags=re.MULTILINE)

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)

            print(f"成功替换 {count} 处内容, 原始模式: {regex_pattern}, 替换为: {substitute}")
            return True

        except FileNotFoundError:
            print(f"文件不存在: {file_name}")
            return False
        except Exception as e:
            print(f"处理文件时发生错误: {e}")
            return False

    def move_content_by_regex(
            self,
            file_name: str,
            source_pattern: str,
            target_pattern: str,
            insert_after_group: int = 0
    ) -> bool:
        """
        将文件中匹配 source_pattern 的内容移动到 target_pattern 指定位置

        参数:
        file_name (str): 文件名（不含路径）
        source_pattern (str): 源内容正则表达式（要移动的内容）
        target_pattern (str): 目标位置正则表达式（需包含分组以精确定位插入点）
        insert_after_group (int):
            - -1: 在整个目标匹配开始前插入
            - 0: 在整个目标匹配结束后插入
            - n>0: 在第n个捕获分组结束后插入（推荐用于精确定位）

        返回:
        bool: 操作是否成功
        """
        try:
            file_path = os.path.join(self.save_dir, file_name)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # 步骤1: 查找并提取源内容（只处理第一个匹配）
            source_match = re.search(source_pattern, content, flags=re.MULTILINE | re.DOTALL)
            if not source_match:
                print(f"❌ 未找到源内容匹配: {source_pattern}")
                return False

            source_text = source_match.group()
            source_start, source_end = source_match.span()

            # 步骤2: 查找目标位置（在原始内容中定位，避免删除后位置偏移）
            target_match = re.search(target_pattern, content, flags=re.MULTILINE | re.DOTALL)
            if not target_match:
                print(f"❌ 未找到目标位置匹配: {target_pattern}")
                return False

            # 步骤3: 计算插入位置（考虑源内容删除对目标位置的影响）
            if insert_after_group == -1:
                insert_pos = target_match.start()
            elif insert_after_group == 0:
                insert_pos = target_match.end()
            else:
                # 验证分组索引有效性
                if insert_after_group > len(target_match.groups()):
                    print(f"❌ 目标匹配仅包含 {len(target_match.groups())} 个分组，无法使用分组 {insert_after_group}")
                    return False
                insert_pos = target_match.end(insert_after_group)

            # 调整插入位置：如果源内容在目标位置之前，删除后目标位置会前移
            if source_end <= insert_pos:
                insert_pos -= (source_end - source_start)

            # 步骤4: 构建新内容（先删除源内容，再在目标位置插入）
            # 先删除源内容（保留原始内容用于位置计算）
            content_without_source = content[:source_start] + content[source_end:]

            # 在调整后的位置插入
            new_content = (
                    content_without_source[:insert_pos] +
                    f" {source_text}" +
                    content_without_source[insert_pos:]
            )

            # 步骤5: 写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)

            # 输出调试信息
            source_preview = source_text[:60].replace('\n', '\\n') + ("..." if len(source_text) > 60 else "")
            print(f"✅ 成功移动内容: '{source_preview}'")
            print(f"   源位置: 第{content[:source_start].count(chr(10)) + 1}行")
            print(f"   目标位置: 第{content[:insert_pos].count(chr(10)) + 1}行 (分组{insert_after_group}后)")
            return True

        except FileNotFoundError:
            print(f"❌ 文件不存在: {file_name}")
            return False
        except Exception as e:
            print(f"❌ 处理文件时发生错误: {type(e).__name__}: {e}")
            return False