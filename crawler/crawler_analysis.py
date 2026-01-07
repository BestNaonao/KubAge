import re
from collections import defaultdict
import os
import argparse
import numpy as np
from docx import Document


def check_docx_files(folder_path, target_text):
    """
    检查文件夹中的docx文件是否包含目标文本并统计后续文本长度
    :param folder_path: docx文件所在文件夹路径
    :param target_text: 要查找的目标文本
    :return: 包含匹配信息和统计结果的数据
    """
    # 初始化结果存储
    results = []
    total_files = 0
    matched_files = 0
    post_lengths = []  # 存储每个匹配项后续文本的长度

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            total_files += 1

            try:
                doc = Document(file_path)
                full_text = '\n'.join([para.text for para in doc.paragraphs])

                # 检查是否包含目标文本
                if target_text in full_text:
                    matched_files += 1

                    # 查找所有目标文本位置
                    start_indices = [m.start() for m in re.finditer(re.escape(target_text), full_text)]

                    for start_idx in start_indices:
                        # 计算目标文本后的内容起始位置
                        post_start = start_idx + len(target_text)
                        # 提取目标文本后的全部内容
                        post_text = full_text[post_start:]

                        # 清理多余换行符
                        cleaned_text = re.sub(r'\n{3,}', '\n\n', post_text)

                        # 记录后续文本长度
                        post_length = len(post_text)
                        post_lengths.append(post_length)

                        # 添加到结果
                        results.append({
                            'filename': filename,
                            'position': start_idx,
                            'post_text': cleaned_text,
                            'post_length': post_length
                        })
                else:
                    print("xxx: " + filename)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                results.append({
                    'filename': filename,
                    'error': str(e)
                })

    # 计算比例
    if total_files > 0:
        percentage = (matched_files / total_files) * 100
    else:
        percentage = 0

    # 计算后续文本长度统计
    length_stats = {}
    if post_lengths:
        length_stats = {
            'max': max(post_lengths),
            'min': min(post_lengths),
            'mean': np.mean(post_lengths),
            'median': np.median(post_lengths),
            'total': sum(post_lengths),
            'count': len(post_lengths)
        }

    return {
        'matches': results,
        'stats': {
            'total_files': total_files,
            'matched_files': matched_files,
            'percentage': percentage,
            'post_length_stats': length_stats
        }
    }


def format_length(length):
    """格式化长度值为易读形式"""
    if length < 1000:
        return f"{length} 字符"
    elif length < 1000000:
        return f"{length / 1000:.1f}千字符"
    else:
        return f"{length / 1000000:.2f}兆字符"


def report_content_check(results):
    """生成文档内容检查报告"""
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("文档内容检查报告")
    report_lines.append("=" * 50)

    # 打印匹配内容和位置
    for match in results['matches']:
        if 'post_text' in match:
            report_lines.append(f"\n📄 文件: {match['filename']}")
            report_lines.append(f"📍 位置: {match['position']}")
            report_lines.append(f"📏 后续文本长度: {format_length(match['post_length'])}")

            # 只显示前200个字符作为预览
            preview = match['post_text'][:200] + ("..." if len(match['post_text']) > 200 else "")
            report_lines.append(f"📝 后续内容预览: \n{preview}")
            report_lines.append('-' * 50)

    # 打印统计信息
    stats = results['stats']
    report_lines.append("\n📊 总体统计:")
    report_lines.append(f"总文件数: {stats['total_files']}")
    report_lines.append(f"包含目标文本的文件数: {stats['matched_files']}")
    report_lines.append(f"占比: {stats['percentage']:.2f}%")

    # 打印后续文本长度统计
    if stats['post_length_stats']:
        len_stats = stats['post_length_stats']
        report_lines.append("\n📏 后续文本长度统计:")
        report_lines.append(f"匹配项数量: {len_stats['count']}")
        report_lines.append(f"最大长度: {format_length(len_stats['max'])}")
        report_lines.append(f"最小长度: {format_length(len_stats['min'])}")
        report_lines.append(f"平均长度: {format_length(len_stats['mean'])}")
        report_lines.append(f"中位长度: {format_length(len_stats['median'])}")
        report_lines.append(f"总长度: {format_length(len_stats['total'])}")
    else:
        report_lines.append("\n⚠️ 未找到匹配项，无法计算后续文本长度统计")

    return "\n".join(report_lines)


# ======================== 日志分析功能 ========================
def analyze_crawler_log(log_lines, save_dir=None):
    """分析爬虫日志文件，提取访问的URL、保存的文件、失败信息等"""
    # 如果传入的是文件路径，则读取文件内容
    if isinstance(log_lines, str) and os.path.isfile(log_lines):
        with open(log_lines, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

    # 解析日志的正则表达式
    start_pattern = re.compile(r'开始爬取: (https?://\S+)')
    saved_pattern = re.compile(r'文档已保存为 (.+\.docx)')
    failed_pattern = re.compile(r'❌ 处理失败 (https?://\S+):')
    progress_pattern = re.compile(r'📊 进度: (\d+)/(\d+)')
    thread_pattern = re.compile(r'线程: (Worker-\d+)')

    # 存储结果的数据结构
    results = {
        'visited_urls': set(),
        'saved_files': set(),
        'failed_urls': set(),
        'url_to_file': {},
        'file_to_urls': defaultdict(list),
        'thread_activities': defaultdict(list),
        'max_visited': 0,
        'max_total': 0
    }

    current_thread = None

    for line in log_lines:
        # 捕获当前线程
        thread_match = thread_pattern.search(line)
        if thread_match:
            current_thread = thread_match.group(1)

        # 捕获开始爬取的URL
        start_match = start_pattern.search(line)
        if start_match:
            url = start_match.group(1)
            results['visited_urls'].add(url)
            if current_thread:
                results['thread_activities'][current_thread].append(('start', url))

        # 捕获保存的文件
        saved_match = saved_pattern.search(line)
        if saved_match:
            filepath = saved_match.group(1)
            filename = os.path.basename(filepath)
            results['saved_files'].add(filename)
            if current_thread:
                results['thread_activities'][current_thread].append(('saved', filename))

        # 捕获失败的URL
        failed_match = failed_pattern.search(line)
        if failed_match:
            url = failed_match.group(1)
            results['failed_urls'].add(url)
            if current_thread:
                results['thread_activities'][current_thread].append(('failed', url))

        # 捕获进度信息
        progress_match = progress_pattern.search(line)
        if progress_match:
            visited = int(progress_match.group(1))
            total = int(progress_match.group(2))
            if total > results['max_total']:
                results['max_total'] = total
                results['max_visited'] = visited

    # 构建URL和文件的映射关系
    for thread, activities in results['thread_activities'].items():
        current_url = None
        for action, value in activities:
            if action == 'start':
                current_url = value
            elif action == 'saved' and current_url:
                results['url_to_file'][current_url] = value
                results['file_to_urls'][value].append(current_url)

    # 分析文件系统中的实际文件（如果提供了保存目录）
    actual_files = set()
    if save_dir and os.path.isdir(save_dir):
        actual_files = set(os.listdir(save_dir))
        actual_files = {f for f in actual_files if f.endswith('.docx')}

    return results, actual_files


def report_crawler_findings(results, actual_files=None):
    """生成并打印爬虫过程分析报告"""
    report = []
    report.append("=" * 50)
    report.append("爬虫日志分析报告")
    report.append("=" * 50)

    # 1. 基本统计
    report.append("\n[基本统计]")
    report.append(f"日志中记录的访问URL数量: {len(results['visited_urls'])}")
    report.append(f"日志中记录的保存文件数量: {len(results['saved_files'])}")
    report.append(f"日志中记录的失败URL数量: {len(results['failed_urls'])}")

    if actual_files is not None:
        report.append(f"文件系统中的实际文件数量: {len(actual_files)}")
    report.append(f"最终进度: {results['max_visited']}/{results['max_total']}")

    # 2. 缺失文件分析（如果有实际文件信息）
    if actual_files is not None:
        missing_in_log = actual_files - results['saved_files']
        missing_in_fs = results['saved_files'] - actual_files

        report.append("\n[文件系统差异]")
        if missing_in_log:
            report.append(f"警告: {len(missing_in_log)}个文件存在于文件系统但未在日志中记录")
            report.append("这些文件可能是之前运行留下的或手动添加的")

        if missing_in_fs:
            report.append(f"严重: {len(missing_in_fs)}个文件在日志中记录但不存在于文件系统")
            for file in missing_in_fs:
                report.append(f"  - {file}")

    # 3. 失败URL分析
    if results['failed_urls']:
        report.append("\n[失败URL列表]")
        for url in results['failed_urls']:
            report.append(f"  - {url}")

    # 4. 文件冲突分析
    conflict_files = {f: urls for f, urls in results['file_to_urls'].items() if len(urls) > 1}

    if conflict_files:
        report.append("\n[文件名冲突警告]")
        report.append(f"发现 {len(conflict_files)} 个文件名被多个URL使用:")
        for filename, urls in conflict_files.items():
            report.append(f"\n文件名: {filename}")
            report.append("对应的URL:")
            for url in urls:
                report.append(f"  - {url}")
    else:
        report.append("\n[文件名冲突检查] 未发现文件名冲突")

    # 5. 未保存URL分析
    unsaved_urls = results['visited_urls'] - set(results['url_to_file'].keys()) - results['failed_urls']

    if unsaved_urls:
        report.append("\n[未保存URL分析]")
        report.append(f"发现 {len(unsaved_urls)} 个URL被访问但未保存:")
        for url in unsaved_urls:
            report.append(f"  - {url}")

        # 尝试找出最后访问的URL
        report.append("\n可能的罪魁祸首(最后访问的URL):")
        last_url = list(unsaved_urls)[-1] if unsaved_urls else None
        report.append(f"  - {last_url}")
    else:
        report.append("\n[未保存URL分析] 所有访问的URL都已保存或标记为失败")

    return "\n".join(report)


def analyze_crawler_errors(log_lines):
    """分析爬虫日志文件中的错误"""
    # 如果传入的是文件路径，则读取文件内容
    if isinstance(log_lines, str) and os.path.isfile(log_lines):
        with open(log_lines, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

    # 错误统计字典结构: {错误类型: {"count": 数量, "urls": [URL列表]}}
    error_stats = defaultdict(lambda: {"count": 0, "urls": set()})

    # 预编译正则表达式提高效率
    error_pattern = re.compile(r"❌ 处理失败 (.*?): (.*)")
    url_extract_pattern = re.compile(r"https?://\S+")

    for line in log_lines:
        # 检查是否为错误行
        match = error_pattern.search(line)
        if not match:
            continue

        # 提取URL和错误信息
        raw_url, error_msg = match.groups()

        # 清理URL（移除可能的尾随标点）
        clean_url = raw_url.strip()
        if clean_url.endswith(('.', ',')):
            clean_url = clean_url[:-1]

        # 标准化错误信息
        normalized_error = error_msg.strip()
        if 'HTTPSConnectionPool' in normalized_error:
            normalized_error = "ConnectionError"
        elif 'Read timed out' in normalized_error:
            normalized_error = "TimeoutError"
        elif '404' in normalized_error:
            normalized_error = "HTTP 404"
        elif 'SSLError' in normalized_error:
            normalized_error = "SSLError"

        # 更新统计信息
        error_stats[normalized_error]["count"] += 1
        error_stats[normalized_error]["urls"].add(clean_url)

    return dict(error_stats)


def save_error_report(stats, output_file):
    """保存错误统计报告到文件"""
    report_lines = []
    report_lines.append("爬虫错误统计分析报告")
    report_lines.append("=" * 50)
    report_lines.append("")

    # 按错误数量降序排序
    sorted_errors = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)

    for error_type, data in sorted_errors:
        report_lines.append(f"错误类型: {error_type}")
        report_lines.append(f"出现次数: {data['count']}")
        report_lines.append(f"相关URL ({len(data['urls'])}个):")

        # 列出相关URL
        for url in data['urls']:
            report_lines.append(f"  - {url}")

        report_lines.append("")
        report_lines.append("-" * 50)
        report_lines.append("")

    with open(output_file, 'w', encoding='utf-8') as report:
        report.write("\n".join(report_lines))


def generate_error_summary(stats):
    """生成错误摘要信息"""
    if not stats:
        return "未发现错误信息"

    summary = []
    summary.append("错误摘要:")
    summary.append(f"{'错误类型':<25} | {'次数':<5} | {'影响URL数量':<10}")
    summary.append("-" * 50)

    for error, data in sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True):
        summary.append(f"{error:<25} | {data['count']:<5} | {len(data['urls']):<10}")

    return "\n".join(summary)


# ======================== 主程序 ========================
def main():
    parser = argparse.ArgumentParser(description='综合日志分析和文档检查工具')
    parser.add_argument('mode', nargs='?', choices=['log', 'content', 'errors'],
                        help='分析模式: log(日志过程分析), content(文档内容检查), errors(错误分析)')

    # 日志分析参数
    parser.add_argument('--log-file', help='爬虫日志文件路径')
    parser.add_argument('--doc-dir', help='文档保存目录（用于文件系统验证）', default='../raw_data')

    # 文档内容检查参数
    parser.add_argument('--target-text', help='文档内容检查的目标文本', default='此页是否对你有帮助？')

    # 错误分析参数
    parser.add_argument('--error-report', help='错误报告输出文件', default='crawler_errors_report.txt')

    # 输出控制
    parser.add_argument('--output', help='结果输出文件', default=None)

    args = parser.parse_args()

    output_file = args.output

    # 根据模式执行不同的分析
    if args.mode == 'log':
        if not args.log_file:
            print("错误: 日志分析需要指定--log-file参数")
            return

        print("\n执行爬取过程分析...")
        with open(args.log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

        results, actual_files = analyze_crawler_log(log_lines, args.doc_dir)
        report = report_crawler_findings(results, actual_files)

    elif args.mode == 'content':
        print("\n执行文档内容检查...")
        results = check_docx_files(args.doc_dir, args.target_text)
        report = report_content_check(results)

    elif args.mode == 'errors':
        if not args.log_file:
            print("错误: 错误分析需要指定--log-file参数")
            return

        print("\n执行错误分析...")
        with open(args.log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

        error_stats = analyze_crawler_errors(log_lines)
        if error_stats:
            save_error_report(error_stats, args.error_report)
            report = generate_error_summary(error_stats)
            report += f"\n详细错误报告已保存至: {args.error_report}"
        else:
            report = "未发现错误信息"

    else:
        print("请指定分析模式: log, content 或 errors")
        return

    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"结果已保存至: {output_file}")
    else:
        print("\n" + report)


if __name__ == '__main__':
    """
    # 文档内容检查
    python crawler_analysis.py content --doc-dir ../raw_data --target-text "目标文本"
    
    # 日志过程分析
    python crawler_analysis.py log --log-file crawler.log --doc-dir ../raw_data
    
    # 错误分析
    python crawler_analysis.py errors --log-file crawler.log
    
    # 输出结果到文件
    python crawler_analysis.py content --doc-dir ../raw_data --output content_report.txt
    """
    main()