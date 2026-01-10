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
        markdown_content = convert_to_markdown(soup)
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

