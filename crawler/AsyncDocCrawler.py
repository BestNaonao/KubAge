import asyncio
import logging
import os
import re
import traceback
from typing import List, Dict, Set
from urllib.parse import urljoin

import html2text
import aiohttp
from bs4 import BeautifulSoup

# 常量保持不变
HEADERS: Dict[str, str] = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Connection": "keep-alive"
}
NO_REF_URL: str = "https://kubernetes.io"
NO_REF_LEN: int = len(NO_REF_URL)
BASE_URL: str = "https://kubernetes.io/zh-cn/docs/"
START_URL: str = BASE_URL + "home/"
ELEMENT_NAME_LIST: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'code']
EXCLUDE_HREF_PREFIXES = ["/contribute", "/blog", "/training", "/careers", "/partners", "/community", "/test"]
REMOVE_TEXT: str = "此页是否对你有帮助"


class AsyncDocCrawler:
    def __init__(self, num_workers: int, save_dir: str):
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.visit_lock = asyncio.Lock()
        self.found: Set[str] = set()
        self.to_visit = asyncio.Queue()
        self.errored: List[str] = []  # 只追加

        # 日志配置
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("crawler_async.log", encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    # ========== 以下业务方法完全保留原逻辑 ==========
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

    def get_unique_filename(self, base_name: str) -> str:
        """生成唯一的文件名，避免冲突"""
        counter = 1
        candidate = base_name
        while os.path.exists(os.path.join(self.save_dir, f"{candidate}.md")):
            candidate = f"{base_name}_{counter}"
            counter += 1
        return candidate

    def get_span_path(self, soup, url: str) -> str:
        # 尝试找到当前页面的导航路径
        a_tag = soup.find('a', attrs={'href': url[NO_REF_LEN:]})
        active_span = a_tag and a_tag.find('span')

        if not active_span:
            page_name = soup.find('h1', attrs={'data-pagefind-weight': '10'})
            return page_name.get_text(strip=True).replace('/', '和') if page_name else "unknown"

        path = [active_span.get_text(strip=True).replace('/', '和')]
        current_li = active_span.find_parent('li')

        while current_li:
            if not (ul := current_li.find_parent('ul')) or not (parent_li := ul.find_parent('li')):
                break
            if not (span := parent_li.select_one('label > a > span')):
                break
            path.insert(0, span.get_text(strip=True).replace('/', '和'))
            current_li = parent_li

        return self.get_unique_filename('_'.join(path))

    @staticmethod
    def convert_to_markdown(title: str, main_content) -> str:
        """将HTML内容转换为Markdown格式"""
        html2text_processor = html2text.HTML2Text()
        html2text_processor.body_width = 0
        html2text_processor.ignore_links = True
        html2text_processor.ignore_images = False
        html2text_processor.ignore_emphasis = False
        html2text_processor.mark_code = True

        for element in main_content.select('.toolbar, .edit-page-link, .feedback--title, .pagehead'):
            element.decompose()

        breadcrumb_nav = main_content.find('nav')
        breadcrumb_str = ""
        if breadcrumb_nav:
            breadcrumb_items = [a.get_text().strip() for a in breadcrumb_nav.select('ol li a')]
            breadcrumb_str = " / ".join(breadcrumb_items)
            breadcrumb_nav.decompose()

        code_block_placeholders = {}

        for i, alert_div in enumerate(main_content.find_all('div', class_=lambda x: x and 'alert ' in x)):
            alert_class = ' '.join(alert_div.get('class'))
            alert_text = html2text_processor.handle(str(alert_div)).strip()
            placeholder = f"<!--ALERT_BLOCK_{i}-->"
            code_block_placeholders[placeholder] = f"\n:::{alert_class}\n{alert_text}\n:::\n"
            alert_div.replace_with(placeholder)

        for i, pre in enumerate(main_content.find_all('pre')):
            code_tag = pre.find('code')
            if code_tag:
                language = code_tag.get('data-lang', '').strip()
                lang_tag = language if language else ''
                code_text = code_tag.get_text().strip()
                placeholder = f"<!--CODE_BLOCK_{i}-->"
                code_block_placeholders[placeholder] = f"\n```{lang_tag}\n{code_text}\n```\n"
                pre.replace_with(placeholder)

        markdown_content = f"# {title}\n\n{breadcrumb_str}\n\n"
        markdown_content += html2text_processor.handle(str(main_content))

        for placeholder, code_block in code_block_placeholders.items():
            markdown_content = markdown_content.replace(placeholder, code_block)

        markdown_content = re.sub(r'```\s*\n', '```\n', markdown_content)
        markdown_content = re.sub(r'\n-{3,}\n', '\n\n', markdown_content)
        markdown_content = re.sub(r'\| :--- ', '| :--- | ', markdown_content)
        markdown_content = markdown_content.split("## 反馈")[0]

        return markdown_content

    async def extract_content(self, session: aiohttp.ClientSession, url: str):
        """提取网页中的内容，包括标题和正文，最后保存"""
        try:
            self.logger.info(f"开始爬取: {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                html = await resp.text()

            soup = BeautifulSoup(html, 'lxml')

            # 提取新链接
            new_urls = self.extract_new_urls(soup, url)
            async with self.visit_lock:
                for u in new_urls:
                    if u not in self.found:
                        self.found.add(u)
                        await self.to_visit.put(u)

            doc_name = self.get_span_path(soup, url)
            main_content = soup.find('main')
            if not main_content:
                self.logger.warning(f"⚠️ 未找到正文内容: {url}")
                return

            title = soup.title.string.replace("| Kubernetes", "").strip()
            markdown_content = self.convert_to_markdown(title, main_content)

            save_path = os.path.join(self.save_dir, f"{doc_name}.md")
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            self.logger.info(f"已保存: {save_path} (字符数: {len(markdown_content)})")

        except Exception as e:
            self.errored.append(url)
            self.logger.error(f"❌ 处理失败 {url}: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1)

    async def worker(self, session: aiohttp.ClientSession):
        while True:
            try:
                url = await asyncio.wait_for(self.to_visit.get(), timeout=10.0)
                await self.extract_content(session, url)
                self.to_visit.task_done()
            except asyncio.TimeoutError:
                self.logger.info("Worker 超时退出，队列可能已空")
                break
            except asyncio.CancelledError:
                self.logger.info("Worker 被取消")
                break
            except Exception as e:
                self.logger.error(f"Worker 异常: {e}")

    async def start(self):
        self.found.clear()
        self.errored.clear()
        self.found.add(START_URL)
        await self.to_visit.put(START_URL)

        # 创建带重试的 aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        async with aiohttp.ClientSession(
            headers=HEADERS,
            timeout=timeout,
            connector=connector
        ) as session:
            workers = [
                asyncio.create_task(self.worker(session))
                for _ in range(self.num_workers)
            ]

            try:
                while True:
                    to_do = self.to_visit.qsize()
                    total = len(self.found)
                    active_count = sum(1 for w in workers if not w.done())

                    if to_do == 0 and active_count == 0:
                        self.logger.info("所有任务完成，没有活跃的worker")
                        break

                    if total > 0:
                        progress = (total - to_do) / total * 100
                        print(
                            f"\r📊 进度: {total - to_do}/{total} ({progress:.1f}%) | "
                            f"队列: {to_do} | 活跃Worker: {active_count} | 错误: {len(self.errored)}",
                            end='', flush=True
                        )
                    await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                self.logger.info("用户中断，正在取消任务...")
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        print("\n" + "=" * 50)
        self.logger.info(f"爬取完成! 总共访问: {len(self.found)} 页面, 错误: {len(self.errored)}")

        if self.errored:
            self.logger.info("----- 以下URL获取失败:")
            for url in self.errored:
                self.logger.info(f"--- {url}")


if __name__ == "__main__":
    crawler = AsyncDocCrawler(num_workers=10, save_dir="../raw_data")
    asyncio.run(crawler.start())