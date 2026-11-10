"""读取仓库根目录的 .env（只在本机，已加入 .gitignore，不会上传）：把 API key 等配置放进环境变量。

规则：KEY=VALUE 一行一个，# 开头是注释，值两侧的引号会去掉；终端里已经 export 的同名变量优先，不会被覆盖。
网关、MCP Server、编排器、评测 / 标注脚本只要 import medagent 就会自动读取；设 MEDAGENT_NO_DOTENV=1 可关闭。
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / ".env"


def load_dotenv(path: str | Path = DEFAULT) -> list[str]:
    """返回本次新设置的变量名（不返回值，避免把 key 打进日志）。"""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
