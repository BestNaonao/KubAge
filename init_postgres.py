import os
import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
LANGGRAPH_DB = os.getenv("LANGGRAPH_DB")

# 管理入口 DB（用于 CREATE DATABASE）
ADMIN_PG_URI = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def init_postgres():
    print("🔍 检查 PostgreSQL 数据库...")
    conn = psycopg.connect(ADMIN_PG_URI, autocommit=True)

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (LANGGRAPH_DB,)
            )
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{LANGGRAPH_DB}"')
                print(f"✅ PostgreSQL 数据库 '{LANGGRAPH_DB}' 创建成功")
            else:
                print(f"✅ PostgreSQL 数据库 '{LANGGRAPH_DB}' 已存在")
    finally:
        conn.close()

    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{LANGGRAPH_DB}"
    )

FULL_PG_URI = init_postgres()

checkpointer_cm = PostgresSaver.from_conn_string(FULL_PG_URI)
checkpointer = checkpointer_cm.__enter__()
checkpointer.setup()
print("✅ PostgreSQL 数据库 初始化完毕")