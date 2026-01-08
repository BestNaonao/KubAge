import logging
import os
import threading
import time
import traceback
from queue import Queue, Empty
from typing import List, Dict
from urllib.parse import urljoin

import html2text
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.html2md_utils import convert_to_markdown, get_span_path


HEADERS: Dict[str, str] = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Connection": "keep-alive"
}
NO_REF_URL: str = "https://kubernetes.io"
BASE_URL: str = "https://kubernetes.io/zh-cn/docs/"
START_URL: str = BASE_URL + "home/"
ELEMENT_NAME_LIST: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'code']
EXCLUDE_HREF_PREFIXES = ["/contribute", "/blog", "/training", "/careers", "/partners", "/community", "/test"]
REMOVE_TEXT: str = "此页是否对你有帮助"


class DocCrawler(object):
    visited: Queue[str]
    to_visit: Queue[str]
    errored: Queue[str]
    session: requests.Session
    lock: threading.Lock
    visited_set: set
    to_visit_set: set
    filename_counter: Dict[str, int]
    active_threads: int
    num_threads: int
    logger: logging.Logger
    save_dir: str
    html2text_processor: html2text.HTML2Text

    def __init__(self, num_threads, save_dir):
        self.visited = Queue()
        self.to_visit = Queue()
        self.errored = Queue()
        self.session = self._create_session()
        self.lock = threading.Lock()
        self.visited_set = set()
        self.to_visit_set = set()
        self.filename_counter = {}
        self.active_threads = 0
        self.num_threads = num_threads
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("crawler.log", encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.html2text_processor = html2text.HTML2Text()
        self.html2text_processor.body_width = 0
        self.html2text_processor.ignore_links = True
        self.html2text_processor.ignore_images = False
        self.html2text_processor.ignore_emphasis = False
        self.html2text_processor.mark_code = True

    @staticmethod
    def _create_session():
        """创建带重试机制的会话"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,  # 最大重试次数
            backoff_factor=1,  # 指数退避因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
            allowed_methods=["GET"]  # 只重试GET请求
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(HEADERS)
        return session

    @staticmethod
    def extract_new_urls(soup, url) -> List[str]:
        """提取页面中的新URL并返回列表"""
        new_urls = []
        parent_elements = soup.select("ul.ul-1")
        for parent_element in parent_elements:
            for link in parent_element.find_all('a', href=True):
                href = link.get('href')
                full_url = urljoin(BASE_URL, link['href'])
                if any(prefix in href for prefix in EXCLUDE_HREF_PREFIXES) or not full_url.startswith(BASE_URL):
                    continue
                if '#' in full_url or full_url == url:
                    continue
                new_urls.append(full_url)
        return new_urls

    def add_new_urls(self, urls: List[str]):
        """线程安全地批量添加新URL"""
        with self.lock:
            for url in urls:
                if url not in self.visited_set and url not in self.to_visit_set:
                    self.to_visit_set.add(url)
                    self.to_visit.put(url)

    def get_unique_filename(self, base_name: str) -> str:
        """生成唯一的文件名，避免冲突"""
        # 使用URL哈希值确保唯一性
        with self.lock:
            # 如果文件名已存在，添加计数器后缀
            if base_name in self.filename_counter:
                self.filename_counter[base_name] += 1
                return f"{base_name}_{self.filename_counter[base_name]}"
            else:
                self.filename_counter[base_name] = 1
                return base_name

    def extract_content(self, url):
        doc_name = None
        main_content = None
        try:
            self.logger.info(f"开始爬取: {url} (线程: {threading.current_thread().name})")
            # 通过会话多次尝试获取网页内容，直到成功
            response = self.session.get(url, timeout=(3.05, 30))
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            # 解析爬取内容，添加新的网页链接，获取文档标题和正文内容
            self.add_new_urls(self.extract_new_urls(soup, url))
            doc_name = self.get_unique_filename(get_span_path(soup, url, NO_REF_URL))
            markdown_content = convert_to_markdown(soup)

            if not markdown_content:
                self.logger.warning(f"⚠️ 未找到正文内容: {url}")
                return

            # 解析保存的文件名，保存到保存目录路径下
            save_path = os.path.join(self.save_dir, f"{doc_name}.md")
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            self.logger.info(f"Markdown文档已保存为 {save_path} (大小: {len(markdown_content)} 字符)")

        except Exception as e:
            with self.lock:
                self.errored.put(url)
            self.logger.error(f"❌ 处理失败 {url}: {str(e)}, traceback: {traceback.format_exc()}")
            with open(f"{doc_name if doc_name else "unknown"}.txt", 'w', encoding='utf-8') as f:
                f.write(str(main_content) if main_content else "")
            time.sleep(3)
        finally:
            with self.lock:
                self.active_threads -= 1

    def worker(self):
        """工作线程函数"""
        while True:
            try:
                url_workon = self.to_visit.get(timeout=10)

                with self.lock:
                    self.visited_set.add(url_workon)
                    self.active_threads += 1

                self.visited.put(url_workon)
                self.extract_content(url_workon)
                self.to_visit.task_done()

            except Empty:
                # 检查是否所有工作都已完成
                with self.lock:
                    if self.active_threads == 0:
                        break
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"工作线程异常: {str(e)}")
                time.sleep(1)

    def start(self):
        """启动多线程爬取"""
        self.clear()
        self.to_visit.put(START_URL)
        self.to_visit_set.add(START_URL)

        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker, name=f"Worker-{i + 1}")
            t.daemon = True
            t.start()
            threads.append(t)

        try:
            while any(t.is_alive() for t in threads):
                with self.lock:
                    visited_count = len(self.visited_set)
                    to_visit_count = self.to_visit.qsize()
                    active_threads = self.active_threads
                    errored_count = self.errored.qsize()
                    total = visited_count + to_visit_count

                if total > 0:
                    progress = visited_count / total * 100
                    # 使用print保持进度条在同一行更新
                    print(f"\r📊 进度: {visited_count}/{total} ({progress:.1f}%) | "
                          f"活动线程: {active_threads}/{self.num_threads} | "
                          f"错误: {errored_count}", end='', flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("\n用户中断，等待线程退出...")

        for t in threads:
            t.join(timeout=5)

        print("\n" + "=" * 50)
        self.logger.info(f"爬取完成! 总共访问: {len(self.visited_set)} 页面, 错误: {self.errored.qsize()}")

        if not self.errored.empty():
            self.logger.info("-----以下URL获取失败:")
            while not self.errored.empty():
                self.logger.info(f"--- {self.errored.get()}")

    def clear(self):
        self.visited.queue.clear()
        self.to_visit.queue.clear()
        self.errored.queue.clear()
        self.visited_set.clear()
        self.to_visit_set.clear()


if __name__ == '__main__':
    crawler = DocCrawler(10, "../raw_data")
    crawler.start()