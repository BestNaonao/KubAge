import json
import os
import platform
import subprocess

from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务
# dependencies 列表可以声明该服务依赖的 Python 包
mcp = FastMCP("os-utils", dependencies=["mcp"])

# 定义工作目录限制，防止 Agent 访问系统敏感目录 (如 /etc, /var 等)
# 在实际生产中，建议将其设置为某个特定的沙箱目录
ALLOWED_ROOT = os.path.abspath("./workspace")
if not os.path.exists(ALLOWED_ROOT):
    os.makedirs(ALLOWED_ROOT)


def validate_path(path: str) -> str:
    """安全检查：确保路径在允许的工作目录内"""
    full_path = os.path.abspath(os.path.join(ALLOWED_ROOT, path))
    if not full_path.startswith(ALLOWED_ROOT):
        raise ValueError(f"Access denied: Path must be within {ALLOWED_ROOT}")
    return full_path


@mcp.tool()
def execute_command(command: str, timeout: int = 30) -> str:
    """
    Execute a shell command on the host system.
    WARNING: Use with caution. Only use for non-interactive commands.

    Args:
        command: The command string to execute (e.g., "echo hello", "ls -la")
        timeout: Execution timeout in seconds (default: 30)
    """
    # 安全拦截：禁止高危命令
    forbidden = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero"]
    if any(bad in command for bad in forbidden):
        return "Error: Command contains forbidden patterns."

    try:
        # 使用 subprocess 执行，捕获输出
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=ALLOWED_ROOT,  # 限制执行目录
            timeout=timeout
        )

        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"

        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the contents of a file from the workspace.

    Args:
        path: Relative path to the file (e.g., "manifests/deploy.yaml")
    """
    try:
        safe_path = validate_path(path)
        if not os.path.exists(safe_path):
            return f"Error: File not found at {path}"

        with open(safe_path, 'r', encoding='utf-8') as f:
            return f.read()
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the workspace. Overwrites if exists.

    Args:
        path: Relative path to the file
        content: The string content to write
    """
    try:
        safe_path = validate_path(path)
        # 确保父目录存在
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)

        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error writing file: {str(e)}"


@mcp.tool()
def get_system_info() -> str:
    """
    Get basic information about the host system (OS, Architecture, Python version).
    Useful for determining compatibility (e.g., which docker image arch to use).
    """
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),  # e.g., x86_64, arm64
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "workspace_root": ALLOWED_ROOT
    }
    return json.dumps(info, indent=2)


@mcp.tool()
def append_environment_variable(key: str, value: str) -> str:
    """
    Append an environment variable to a '.env' file in the workspace.
    This does NOT change the current process env, but helps persistence for future commands
    if they load this file.
    """
    try:
        env_path = validate_path(".env")
        # 简单的追加逻辑
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"\n{key}={value}")
        return f"Appended {key} to .env file"
    except Exception as e:
        return f"Error saving env var: {str(e)}"


if __name__ == "__main__":
    # 启动 MCP 服务器
    print(f"Starting OS Utils MCP Server...")
    print(f"Workspace restricted to: {ALLOWED_ROOT}")
    mcp.run()