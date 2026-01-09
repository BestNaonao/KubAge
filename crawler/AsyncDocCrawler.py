import asyncio
import traceback
from typing import override

import aiohttp

from crawler.K8sCrawler import K8sCrawler


class AsyncDocCrawler(K8sCrawler):
    def __init__(self, num_workers: int, save_dir: str):
        super().__init__(num_workers, save_dir)
        self.lock = asyncio.Lock()
        self.to_visit = asyncio.Queue()

    async def worker(self, session: aiohttp.ClientSession):
        url = ""
        while True:
            try:
                url = await asyncio.wait_for(self.to_visit.get(), timeout=10.0)
                self.logger.info(f"开始爬取: {url}")
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    html = await resp.text()

                new_urls, span_name, markdown_content = self.parse_html(html, url)

                async with self.lock:
                    for u in new_urls:
                        if u not in self.found:
                            self.found.add(u)
                            await self.to_visit.put(u)
                    file_name = self.get_unique_filename(span_name)
                self.save(markdown_content, url, file_name)
                self.to_visit.task_done()
            except asyncio.TimeoutError:
                self.logger.info("Worker 超时退出，队列可能已空")
                break
            except asyncio.CancelledError:
                self.logger.info("Worker 被取消")
                break
            except Exception as e:
                if url:
                    self.errored.append(url)
                self.logger.error(f"❌ 处理失败 {url}: {str(e)}, traceback: {traceback.format_exc()}")
                await asyncio.sleep(1)

    async def start(self):
        self.initialize()
        await self.to_visit.put(self.start_url)

        # 创建带重试的 aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        async with aiohttp.ClientSession(
            headers=self.headers,
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

                    self.show_progress(total - to_do, total, active_count)

                    await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                self.logger.info("用户中断，正在取消任务...")
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        self.quit_print()

    @override
    def run(self):
        asyncio.run(self.start())


if __name__ == "__main__":
    crawler = AsyncDocCrawler(num_workers=10, save_dir="../raw_data")
    crawler.run()