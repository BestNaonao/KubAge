import re
from urllib.parse import urljoin

import html2text
from bs4 import BeautifulSoup, Tag


SPECIAL_HEADER_REG = ["概要", "摘要", "清理", "我何时该使用"]

def _normalize_tables(soup):
    """
    将 table 中的 td / th 内容规范化：
    - 合并多个子节点为单一 HTML
    - 使用 <br> 表示单元格内逻辑换行
    - 将 ul/ol 列表转换为单行文本，用 <br> 分隔列表项
    - 将 pre 元素内容转换为单行文本，用 <br> 分隔各行
    """
    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            # 1. 处理 ul/ol 列表元素
            for list_elem in cell.find_all(["ul", "ol"]):
                # 确定列表项的前缀符号
                list_type = list_elem.name  # 'ul' 或 'ol'

                # 收集处理后的列表项文本
                list_items_text = []
                for li in list_elem.find_all("li"):
                    # 获取 li 的文本内容，去除多余的空白字符
                    li_text = li.get_text(strip=True)

                    # 去掉 ::marker 伪元素文本（如果存在）
                    if li_text.startswith("::marker"):
                        li_text = li_text[len("::marker"):].strip()

                    # 根据列表类型添加前缀
                    if list_type == "ul":
                        # 无序列表，添加 "* "
                        if li_text:
                            list_items_text.append(f" * {li_text}")
                    else:  # ol
                        # 有序列表，添加序号。注意：这里简化处理，不考虑嵌套和多级列表
                        if li_text:
                            # 获取 li 在列表中的索引
                            li_index = list(list_elem.find_all("li")).index(li) + 1
                            list_items_text.append(f" {li_index}. {li_text}")

                # 将列表项用 <br> 连接
                replacement_text = "&lt;br&gt;".join(list_items_text)

                # 用处理后的文本替换整个列表元素
                list_elem.replace_with(replacement_text)

            # 2. 处理 pre 元素
            for pre_elem in cell.find_all("pre"):
                # 获取 pre 元素的文本内容
                pre_text = pre_elem.get_text()
                cleaned_text = pre_text.strip()

                # 将换行符替换为 <br>
                lines = cleaned_text.splitlines()
                # 过滤掉空行
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                # 用 <br> 连接非空行
                replacement_text = "&lt;br&gt;".join(non_empty_lines)

                # 用处理后的文本替换整个 pre 元素
                pre_elem.replace_with(replacement_text)

            # 3. 处理 <br> 标签（包括原始的以及从 pre 转换添加的）
            for br_tag in cell.find_all("br"):
                br_tag.replace_with("&lt;br&gt;")

def _inject_topology_info(soup: Tag, base_url: str):
    """
    显式注入拓扑信息：
    1. 为所有带 id 的 Header 注入入口标记 [HLINK: base_url#id]
    2. 为所有带 href 的 Anchor 注入出口标记 [HLINK: absolute_url]
    """

    # 1. 处理入口 (Entry Points): Headers with IDs
    # 遍历 h1-h6
    for header in soup.find_all(re.compile(r'^h[1-6]$')):
        header_id = header.get('id')
        if header_id:
            # 构建完整入口 URL
            # 注意：base_url 应该是不带 #fragment 的页面 URL
            entry_url = urljoin(base_url, f"#{header_id}")

            # 构造注入文本，将标记追加到标题内容的末尾
            # 使用 append 而不是 replace_with，可以保留标题内可能存在的 <code> 等格式
            header.append(f" [HLINK: {entry_url}]")

    # 2. 处理出口 (Exit Points): Anchors with href
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']

        # 过滤掉无效链接、JS脚本、锚点自身链接(可选)等
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
            continue

        # 过滤掉标题旁边的 "aria-hidden" 永久链接图标，避免重复干扰
        if a_tag.get('aria-hidden') == 'true':
            continue

        # 将相对路径转换为绝对路径
        full_url = urljoin(base_url, href)

        # 构造注入文本，将标记追加到链接文本的末尾
        a_tag.append(f"[HLINK: {full_url}]")

def convert_to_markdown(soup: BeautifulSoup, url: str) -> str:
    """将HTML内容转换为Markdown格式"""
    main_content = soup.find('main')
    if not main_content:
        return ""
    title = soup.title.string.replace("| Kubernetes", "").strip()

    html2text_processor = html2text.HTML2Text()
    html2text_processor.body_width = 0
    html2text_processor.ignore_links = True
    html2text_processor.ignore_images = False
    html2text_processor.ignore_emphasis = False
    html2text_processor.mark_code = True

    # 预处理HTML内容
    # 移除不需要的元素
    for element in main_content.select('.toolbar, .edit-page-link, .feedback--title, .pagehead, #third-party-content-disclaimer'):
        element.decompose()

    # 修复错误嵌套的alert div（在移除其他元素后立即处理）
    fix_misnested_alerts(main_content)
    # 修复标题层级问题
    fix_header_hierarchy(main_content, soup)

    # 提取并处理面包屑导航
    breadcrumb_nav = main_content.find('nav')
    breadcrumb_str = ""
    if breadcrumb_nav:
        breadcrumb_items = [a.get_text().strip() for a in breadcrumb_nav.select('ol li a')]
        breadcrumb_str = " / ".join(breadcrumb_items)
        breadcrumb_nav.decompose()

    # 创建占位符映射表
    code_block_placeholders = {}

    # 处理提醒框
    for i, alert_div in enumerate(main_content.find_all('div', class_=lambda x: x and 'alert ' in x)):
        alert_class = ' '.join(alert_div.get('class'))
        alert_text = html2text_processor.handle(str(alert_div)).strip()
        placeholder = f"<!--ALERT_BLOCK_{i}-->"
        code_block_placeholders[placeholder] = f"\n:::{alert_class}\n{alert_text}\n:::\n"
        alert_div.replace_with(placeholder)

    # 处理外部代码块
    for i, pre in enumerate(main_content.find_all('pre')):
        code_tag = pre.find('code')
        if code_tag:
            language = code_tag.get('data-lang', '').strip()
            lang_tag = language if language else ''
            code_text = code_tag.get_text().strip()
            placeholder = f"<!--CODE_BLOCK_{i}-->"
            code_block_placeholders[placeholder] = f"\n```{lang_tag}\n{code_text}\n```\n"
            pre.replace_with(placeholder)

    # 处理表格
    _normalize_tables(main_content)

    # 注入 HLINK 拓扑信息，传入当前页面的 URL 作为 base_url
    _inject_topology_info(main_content, url)

    # 添加标题前缀和网址
    markdown_content = f"# {title}\n[HLINK: {url}]\n{breadcrumb_str}\n\n"
    markdown_content += html2text_processor.handle(str(main_content))

    # 后处理Markdown，将占位符替换为实际的代码块
    for placeholder, code_block in code_block_placeholders.items():
        markdown_content = markdown_content.replace(placeholder, code_block)

    # 修复代码块格式
    markdown_content = re.sub(r'```\s*\n', '```\n', markdown_content)
    # 移除多余的水平线
    markdown_content = re.sub(r'\n-{3,}\n', '\n\n', markdown_content)
    # 修复表格对齐
    markdown_content = re.sub(r'\| :--- ', '| :--- | ', markdown_content)
    # 修复html的转义<br>转回markdown的<br>
    markdown_content = re.sub(r'&lt;br&gt;', '<br>', markdown_content)
    # 移除反馈以及后续内容
    markdown_content = markdown_content.split("## 反馈")[0]

    return markdown_content

def fix_misnested_alerts(main_content: BeautifulSoup):
    """
    修复错误嵌套的alert div：将alert内部的第一个h2及其后续内容移出到alert之后
    """
    # 查找所有可能错误嵌套的alert div
    for alert_div in main_content.find_all('div', class_=lambda x: x and 'alert' in x.split()):
        # 查找alert内部的第一个h2（或h1/h3等标题，根据需要扩展）
        header_tag = alert_div.find(['h1', 'h2', 'h3'])

        # 如果没有找到标题，或者标题是第一个子元素（正常情况），则跳过
        if not header_tag or header_tag == alert_div.contents[0]:
            continue

        # 获取需要移出的内容：从标题开始到alert结束的所有兄弟元素
        elements_to_move = [header_tag] + header_tag.find_next_siblings()

        # 将这些元素从alert中移除，然后按原始顺序将元素插入到alert之后
        for elem in elements_to_move:
            elem.extract()
        elements_to_move.reverse()
        for elem in elements_to_move:
            alert_div.insert_after(elem)

def fix_header_hierarchy(main_content: Tag, soup: BeautifulSoup):
    """
    修复标题跨级问题：
    - 从第一个<h1>开始处理所有后续标题
    - 基于文档结构关系（而非数值）确定标题层级
    - 正确处理多个h1的情况（后续h1视为新章节起点）
    """
    first_h1 = main_content.find('h1')
    if not first_h1 or first_h1 is None:
        return

    headers: list[Tag] = [first_h1] + first_h1.find_next_siblings(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    if len(headers) < 2:
        return

    # 步骤1: 构建父级索引数组
    # parents[i] = headers[i] 的父级标题在headers中的索引，-1表示父级是第一个h1
    n = len(headers)
    parents = [-1] * n

    # 使用栈跟踪可能的父级，栈中存储 (index, original_level) 元组，按层级从小到大排列
    stack: list[tuple[int, int]] = []

    for i, header in enumerate(headers):
        current_level = int(header.name[1])  # 从标签名提取层级数字

        # 弹出栈中所有层级大于等于当前标题的元素
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        # 确定父级
        if stack:
            parents[i] = stack[-1][0]   # 栈顶元素是最近的、层级小于当前标题的父级
        else:
            parents[i] = -1 # 没有合适的父级，使用 h1 作为父级

        # 将当前标题压入栈
        stack.append((i, current_level))

    # 第二步：根据父级关系计算正确的层级
    # 父级下标为 -1 的标题层级为 1 (h1), 其他标题层级 = 父级层级 + 1
    corrected_levels = [0] * n

    for i in range(n):
        if parents[i] == -1:
            # 父级是 h1，层级为 2
            corrected_levels[i] = 1
        else:
            # 父级是 headers_after_h1 中的某个标题
            corrected_levels[i] = corrected_levels[parents[i]] + 1

    # 应用修正后的层级
    for i in range(n):
        header = headers[i]
        header_text = header.get_text(strip=True)
        new_tag_name = f'h{corrected_levels[i]}'
        if header.name != new_tag_name:
            if any(phrase in header for phrase in SPECIAL_HEADER_REG):
                strong_tag = soup.new_tag('strong')
                strong_tag.string = header_text
                header.replace_with(strong_tag)
            else:
                header.name = new_tag_name

def get_span_path(soup, url: str, no_ref_url) -> str:
    # 尝试找到当前页面的导航路径
    a_tag = soup.find('a', attrs={'href': url[len(no_ref_url):]})
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
        # 将层级名称插入路径最前面
        path.insert(0, span.get_text(strip=True).replace('/', '和'))
        current_li = parent_li  # 继续向上查找

    return '_'.join(path)