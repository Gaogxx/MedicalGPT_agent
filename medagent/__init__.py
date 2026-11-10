"""Medical-GPT Agent runtime.

分层：
  router/      RouteLLM 路由（本地 MedicalGPT ↔ 云端强医疗模型）+ 安全优先策略
  mcp_servers/ 医疗工具 MCP Server（替代原 DSH tool registry）
  mcp_client   MCP 客户端（工具发现 / 调用，经容错层）
  rag/         查询改写 → 多路召回 → 混合检索(RRF) → 重排 → 压缩 → 缓存
  resilience/  错误归因、分类重试、Redis 熔断、幂等、Saga 补偿、断点续跑、乐观并发
  verify/      结果校验（规则 + 轻量判别模型 / Jev 类型化判别）拦截静默失败
  context      上下文压缩（临床关键事实槽位不压缩）
  agent        Router–Worker 编排（参考 DeepSeek Super-agent）
"""

__version__ = "0.2.0"

import os as _os

if _os.environ.get("MEDAGENT_NO_DOTENV") != "1":
    from .envfile import load_dotenv as _load_dotenv
    _load_dotenv()          # 仓库根目录的 .env（本机配置，不入库），已 export 的变量优先
