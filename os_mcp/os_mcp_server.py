import asyncio
import json
import locale
import os
import platform
import shlex

from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务
os_mcp_server = FastMCP("os-utils", dependencies=["mcp"])

# 定义工作目录限制，防止 Agent 访问系统敏感目录 (如 /etc, /var 等)
# 在实际生产中，建议将其设置为某个特定的沙箱目录
ALLOWED_ROOT = os.path.abspath("./workspace")
if not os.path.exists(ALLOWED_ROOT):
    os.makedirs(ALLOWED_ROOT)

IS_SYSTEM_WINDOWS = platform.system() == "Windows"


def validate_path(path: str) -> str:
    """同步辅助函数：路径检查（CPU密集型，极快，无需异步）"""
    full_path = os.path.abspath(os.path.join(ALLOWED_ROOT, path))
    if not full_path.startswith(ALLOWED_ROOT):
        raise ValueError(f"Access denied: Path must be within {ALLOWED_ROOT}")
    return full_path

def decode_bytes(data: bytes) -> str:
    """尝试使用多种编码解码字节流"""
    if not data:
        return ""
    # 1. 优先尝试 UTF-8 (因为很多现代 CLI 工具如 git, docker, python 默认输出 UTF-8)
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        # 2. 如果失败，尝试系统默认编码 (Windows 下通常是 cp936/GBK)
        try:
            return data.decode(locale.getpreferredencoding())
        except UnicodeDecodeError:
            # 3. 如果还不行，强制使用 UTF-8 并替换错误字符 (兜底)
            return data.decode('utf-8', errors='replace')


@os_mcp_server.tool()
async def execute_command(command: str, timeout: int = 60) -> str:
    """
    Execute a shell command asynchronously on the host system.

    WARNING: Use with caution. Only use for non-interactive commands.
    Note: Pipes (|) and redirects (>) are NOT supported due to security restrictions (shlex usage).

    Platform Specifics:
    - Linux/macOS: Executes directly.
    - Windows: The command is automatically wrapped in "cmd /c" to support built-ins
      (like 'dir', 'echo'). You can also explicitly run "powershell <command>"
      or any executable in the system PATH (e.g., "kubectl", "python").

    Args:
        command: The command string to execute (e.g., "ls -la" (Linux), "dir" (Windows), "python --version").
        timeout: Execution timeout in seconds. Defaults to 60.

    Returns:
        str: A human-readable combined string report containing:
             - Exit Code
             - STDOUT (Standard Output)
             - STDERR (Standard Error)
    """
    forbidden = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero"]
    if any(bad in command for bad in forbidden):
        return "Error: Command contains forbidden patterns."

    try:
        # 1. 使用 shlex 解析命令字符串
        args = shlex.split(command, posix=not IS_SYSTEM_WINDOWS)
        if not args:
            return "Error: Empty command"
        if IS_SYSTEM_WINDOWS:
            args = ["cmd", "/c"] + args

        # 2. 使用 asyncio 创建子进程（非阻塞）
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ALLOWED_ROOT,
            # env=... # 如果需要传递环境变量可以在此添加
        )

        # 3. 等待结果，带超时控制
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                # 强制等待子进程完全退出并回收资源，防止僵尸进程和管道泄漏
                await process.wait()
            except ProcessLookupError:
                pass
            return f"Error: Command timed out after {timeout} seconds"

        # 4. 解码输出
        stdout = decode_bytes(stdout_bytes)
        stderr = decode_bytes(stderr_bytes)
        return_code = process.returncode

        output = f"Exit Code: {return_code}\n"
        if stdout:
            output += f"STDOUT:\n{stdout}\n"
        if stderr:
            output += f"STDERR:\n{stderr}\n"

        return output

    except ValueError as e:
        # shlex 解析错误（如引号不匹配）
        return f"Error parsing command: {str(e)}"
    except FileNotFoundError as e:
        return f"Error: Executable File not found: {str(e)}."
    except Exception as e:
        return f"Error executing command: {str(e)}"


@os_mcp_server.tool()
async def read_file(path: str) -> str:
    """
    Read the contents of a file from the workspace.

    Args:
        path: Relative path to the file within the workspace (e.g., "manifests/deploy.yaml").

    Returns:
        str: The text content of the file.
             Returns an error message starting with "Error:" if the file does not exist,
             cannot be read, or executes outside the allowed workspace.
    """
    try:
        # 路径验证是极快的 CPU 操作，不需要 await
        safe_path = validate_path(path)

        if not os.path.exists(safe_path):
            return f"Error: File not found at {path}"

        # 文件 I/O 是阻塞的，使用 to_thread 放入线程池运行
        def _read():
            with open(safe_path, 'r', encoding='utf-8') as f:
                return f.read()

        content = await asyncio.to_thread(_read)
        return content
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error reading file: {str(e)}"


@os_mcp_server.tool()
async def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the workspace. Overwrites if exists.

    Args:
        path: Relative path to the file (e.g., "configs/app-config.json").
              Directories will be created automatically if they don't exist.
        content: The string content to write to the file.

    Returns:
        str: A success message indicating the number of characters written,
             or an error message if the operation fails.
    """
    try:
        safe_path = validate_path(path)

        def _write():
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(content)

        # 同样将写操作放入线程池
        bytes_written = await asyncio.to_thread(_write)
        return f"Successfully wrote {bytes_written} characters to {path}"
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error writing file: {str(e)}"


@os_mcp_server.tool()
async def get_system_info() -> str:
    """
    Get basic information about the host system.

    Useful for determining compatibility (e.g., checking OS type or architecture
    before pulling docker images).

    Returns:
        str: A JSON-formatted string containing:
             - system: OS name (e.g., Linux, Darwin)
             - release: OS version
             - machine: Architecture (e.g., x86_64, arm64)
             - workspace_root: The absolute path of the allowed workspace directory.
    """
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "workspace_root": ALLOWED_ROOT
    }
    return json.dumps(info, indent=2)


@os_mcp_server.tool()
async def append_environment_variable(key: str, value: str) -> str:
    """
    Append an environment variable to a '.env' file in the workspace.

    This does NOT change the current process environment immediately, but preserves
    variables in a file that can be loaded by other tools or future sessions.

    Args:
        key: The environment variable name (e.g., "KUBE_NAMESPACE").
        value: The value to assign.

    Returns:
        str: A success message confirming the variable was appended,
             or an error message if the file operation fails.
    """
    try:
        env_path = validate_path(".env")

        def _append():
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\n{key}={value}")

        await asyncio.to_thread(_append)
        return f"Appended {key} to .env file"
    except Exception as e:
        return f"Error saving env var: {str(e)}"


if __name__ == "__main__":
    import logging
    logging.getLogger("mcp.server").setLevel(logging.WARNING)

    # FastMCP.run() 内部会处理 Event Loop
    os_mcp_server.run()