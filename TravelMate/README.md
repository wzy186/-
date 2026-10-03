# TravelMate — AI 智能出行助手

<p align="center">
  <img src="static/tokyo.svg" width="120" alt="TravelMate">
</p>

<p align="center">
  <strong>基于图编排 ReAct Agent 的全栈 AI 出行助手</strong><br>
  StateGraph · Guardrails · HITL · MCP Tools · RAG · Multi-User
</p>

---

## 项目简介

TravelMate 是一个功能完整的 AI 出行助手，采用 **图编排 Agent 架构**（类 LangGraph），实现了 ReAct 推理循环、Human-in-the-Loop 操作确认、输入/输出安全护栏、MCP 兼容工具协议、RAG 知识检索等现代 Agent 核心能力。

用户通过自然语言对话即可完成行程规划、航班酒店预订、天气查询、汇率换算、翻译等全流程操作，无需在多个 App 之间切换。

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | Streamlit | Python 全栈，快速构建交互式 UI |
| **Agent 引擎** | 自建 StateGraph | 图编排 + 条件路由 + 中断恢复，类 LangGraph |
| **安全护栏** | 自建 Guardrails | 输入/输出双重校验，Prompt 注入检测 + PII 警告 |
| **工具协议** | MCP 兼容 | 17 个工具统一 JSON Schema 注册，可跨框架复用 |
| **LLM** | OpenAI-compatible API | 支持任意 OpenAI 兼容接口 + 本地 Mock |
| **RAG** | TF-IDF + BM25 混合检索 | 纯 Python，无外部依赖，10 城市 114 个知识切片 |
| **地图** | 高德地图 API | 路线规划 / 周边搜索 / 地理编码 / 行政区划 |
| **存储** | JSON 文件持久化 | 多用户隔离，轻量无数据库依赖 |
| **后端 API** | FastAPI | 可选 REST API 层，自动生成 OpenAPI 文档 |
| **数据模型** | Pydantic v2 + SQLAlchemy | 类型安全 + ORM |

## 核心架构（动态多 Agent：Plan-and-Execute）

```
用户输入 → Streamlit UI → 顶层编排图
                              │
                 ┌────────────┴──────────────────┐
                 │                               │
          顶层通用节点                     Supervisor 调度 Agent（Planner）
          ├─ load_context（画像+RAG）      ├─ LLM 生成分阶段执行计划 {"stages":[[...],...]}
          ├─ guardrail_input（护栏）       ├─ 同阶段任务无依赖依次分发 / 依赖任务分阶段
          ├─ guardrail_output（护栏）      ├─ Mock 关键词规划（无 Key 兜底）
          └─ finalize（多回答按Agent分节合并）└─ 接收子 Agent 的 handoff 动态追加阶段
                                                 │ 按计划派发
              ┌──────────────┬───────────────────┼──────────────────┐
              │              │                   │                  │
        route_agent     ticket_agent       travel_agent         qa_agent
        路线规划Agent    票务Agent           行程规划Agent        问答Agent
        ├─ route        ├─ train           ├─ weather(高德实时)  ├─ RAG 检索
        ├─ nearby       ├─ flight          ├─ hotel             └─ 知识问答
        ├─ geocode      ├─ book_flight     ├─ attraction
        └─ district     └─ book_train_     ├─ budget/exchange/
        (高德 REST+MCP)    ticket(HITL)       translate
                        (12306 MCP 真实余票) └─ book_hotel 等(HITL)
              │
   每个子 Agent = 独立 ReAct 子图（react_llm → parse_tools → react_exec 循环，
   最多 3 轮；可 [call:handoff] 把超职责请求动态移交回调度器；
   操作类工具触发 HITL 中断冒泡到顶层，确认后恢复）
              │
   ┌──────────┴──────────────┐
   MCP 统一工具注册表          RAG 知识库 / 记忆系统
   ├─ 本地 12 查询 + 7 操作    ├─ TF-IDF + BM25
   └─ 外部 MCP Server 动态     └─ 10 城市 114 切片
      注册（高德 15 + 12306 8）
```

### 动态多 Agent 机制

- **Plan-and-Execute**：Supervisor 由 LLM 输出 `{"stages": [[任务,...], ...]}`，
  阶段内任务无依赖，阶段间表达依赖（如"先查天气→再按天气做行程"自动排两阶段）
- **handoff 动态移交**：子 Agent 执行中发现超出职责的子请求（如路线Agent遇到订票），
  调用 `[call:handoff] {"agent":"ticket_agent","instruction":"..."}` 交回调度器，
  运行时追加执行阶段，无需重启或预注册
- **状态增量合并**：多分支各自在状态克隆（`AgentState.clone`，容器级浅拷贝）上执行，
  完成后按 fork 点只合并增量（thinking/tool_results/trace/replies 等），
  杜绝继承数据翻倍
- **多格式工具调用解析**：兼容 `[call:tool] {json}` 约定格式、DeepSeek DSML 原生标记、
  通用 `<invoke>`/`<function>` 标记
- **诚实标注**：未接真实数据源的工具结果带 `data_source` 字段，Agent 回答时告知用户

### 外部 MCP Server 接入

在 `.env` 中配置即可自动发现并注册远端工具（连接失败自动降级到本地工具）：

```bash
# 高德官方 MCP（Streamable HTTP）
MCP_AMAP_MODE=http
MCP_AMAP_URL=https://mcp.amap.com/mcp?key=你的高德Key

# 12306 火车票 MCP（stdio，需 Node.js）
MCP_12306_MODE=stdio
MCP_12306_COMMAND=npx -y 12306-mcp
```

远端工具以 `mcp_<server>_<tool>` 命名注册进统一工具注册表，票务/路线 Agent 会自动将其纳入专属工具白名单。

## 项目结构

```
TravelMate/
├── app.py                        # Streamlit 主界面（登录/侧边栏/22页面路由）
├── core/                         # Agent 核心引擎
│   ├── agent.py                  # 多 Agent 编排入口（process/resume，兼容旧接口）
│   ├── graph.py                  # StateGraph 引擎（节点/条件路由/子图嵌入/fan-out分发/增量合并/中断恢复）
│   ├── agents/                   # 多 Agent 编排层
│   │   ├── supervisor.py         # Supervisor：Plan-and-Execute 分阶段调度 + handoff 接收
│   │   ├── base_nodes.py         # 专家子 Agent 工厂（可配置 ReAct 子图）
│   │   ├── configs.py            # 4 个专家 Agent 配置（路线/票务/行程/问答）
│   │   └── nodes.py              # 顶层通用节点（上下文/护栏/收尾）
│   ├── mcp.py                    # MCP 统一工具注册表（本地 + 远端动态注册）
│   ├── mcp_client.py             # 通用 MCP 客户端（stdio + Streamable HTTP）
│   ├── guardrails.py             # 输入/输出护栏（注入检测/内容安全/PII警告）
│   ├── llm.py                    # LLM 客户端（OpenAI + Mock + 3城市×3档次数据）
│   ├── prompts.py                # Prompt 模板
│   ├── memory.py                 # 记忆系统（用户画像 + 会话 + 旅行记录，多用户隔离）
│   └── rag.py                    # RAG 引擎（TF-IDF + BM25，10城市114切片）
├── tools/                        # 工具层
│   ├── base.py                   # 工具基类
│   ├── weather.py                # 天气查询（国内走高德实时，海外降级 Mock 并标注来源）
│   ├── flight.py                 # 航班查询（多航司比价 + 机型/餐食/WiFi）
│   ├── train.py                  # 火车票查询（本地 Mock；12306 MCP 已接入时 Agent 优先用真实余票）
│   ├── hotel.py                  # 酒店推荐（星级/评分/设施/区域指南）
│   ├── exchange.py               # 汇率换算（12种货币）
│   ├── translate.py              # 多语言翻译（8种语言）
│   ├── attraction.py             # 景点推荐（RAG 增强）
│   ├── budget_tool.py            # 预算分配（7分类 + 日均 + 超支预警）
│   └── amap.py                   # 高德地图（路线/周边/编码/行政区划）
├── utils/
│   └── storage.py                # 持久化存储（多用户隔离：登录/预订/收藏/提醒/钱包）
├── data/
│   └── destinations/             # 10 城市知识文件（各 10KB+）
│       ├── tokyo.md / paris.md / bangkok.md / bali.md / london.md
│       ├── seoul.md / newyork.md / sydney.md / dubai.md / rome.md
├── static/                       # 城市 SVG 卡片图（渐变色 + 城市图标）
├── api/                          # FastAPI 后端（可选）
│   ├── main.py
│   └── routes/ (chat.py, trips.py)
├── models/
│   ├── database.py               # SQLAlchemy 模型
│   └── schemas.py                # Pydantic 模型
├── requirements.txt
├── .env.example
└── .gitignore
```

## 功能模块

### 8 大功能模块，22 个页面

| 模块 | 页面 | 能力 |
|------|------|------|
| **AI 对话** | AI 助手 | 自然语言交互，自动调用工具，操作确认（HITL），执行追踪 |
| **行程规划** | 行程规划 / 方案对比 | AI 生成详细行程，3档对比（穷游/舒适/豪华），导出 MD/JSON |
| **出行服务** | 航班 / 酒店 / 景点 / 地图 / 天气 | 查询+预订一体化，地图可视化路线，7日天气预报 |
| **预算管理** | 预算 / 汇率 / AA 分摊 | 智能预算分配，实时记账超支预警，12种货币换算，多人AA |
| **翻译助手** | 翻译 | 8种语言，6种场景翻译卡，短语收藏本 |
| **旅行工具** | 清单 / 签证 | 自动生成打包清单（可勾选），签证材料进度追踪 |
| **知识记录** | 攻略 / 日记 / 知识库 / SOS | AI 生成攻略/日记，语义搜索知识库，紧急求助卡 |
| **个人中心** | 个人主页 / 收藏 / 提醒 / 设置 | 偏好画像，钱包管理，多用户登录注册 |

### Agent 核心特性

| 特性 | 实现方式 |
|------|---------|
| **图编排 (StateGraph)** | 8 节点有向图 + 条件边，支持分支/重试/中断恢复 |
| **Human-in-the-Loop** | 操作类工具（订票/订酒店等）自动暂停，用户确认后执行 |
| **Guardrails 护栏** | 输入：Prompt 注入检测 + 有害内容拦截 + PII 警告；输出：安全校验 |
| **MCP 工具协议** | 17 工具统一 JSON Schema，可跨框架复用 |
| **Tracing 追踪** | 每步记录节点名/耗时/状态变更，可视化展示 |
| **多用户隔离** | 登录注册系统，每个用户独立的预订/收藏/钱包/提醒 |
| **RAG 知识增强** | TF-IDF + BM25 混合检索，10 城市 114 个知识切片 |

## 快速开始

### 环境要求

- Python 3.11+
- pip

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/wzy186/-.git
cd TravelMate

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用（Mock 模式，无需 API Key）
streamlit run app.py
```

浏览器打开 http://localhost:8501，注册账号即可使用。

### 配置真实 AI（可选）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# LLM_API_KEY=sk-xxx
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
# AMAP_API_KEY=你的高德地图Key（可选）
```

| 变量 | 说明 | 必填 |
|------|------|------|
| `LLM_API_KEY` | OpenAI 兼容 API Key | 否（不填则用 Mock） |
| `LLM_BASE_URL` | API 地址 | 否（默认 OpenAI） |
| `LLM_MODEL` | 模型名 | 否（默认 gpt-4o-mini） |
| `AMAP_API_KEY` | 高德地图 API Key | 否（不填则用 Mock） |

## 使用说明

### 基本流程

1. **注册/登录** — 首次访问需注册，每个用户数据独立
2. **AI 对话** — 直接在 AI 助手中用自然语言交互：
   - "帮我规划7月去东京5天的行程，预算1.5万"
   - "帮我订北京到东京最便宜的航班"
   - "东京这周天气怎么样？"
   - "提醒我7月15号办签证"
3. **操作确认** — 订票、订酒店等操作会弹出确认框，确认后才执行
4. **查看追踪** — 每次对话可展开查看 Agent 执行步骤和耗时

### 页面导航

左侧边栏分 5 组导航：
- **发现**：主页 / 行程规划 / 方案对比
- **出行服务**：航班 / 酒店 / 景点 / 地图 / 天气
- **工具箱**：预算 / 汇率 / 翻译 / 清单 / 签证 / AA 分摊
- **记录**：攻略 / 日记 / 知识库 / SOS
- **我的**：个人主页 / AI 助手 / 收藏 / 提醒 / 设置

### Mock 模式

不配置 API Key 时，所有功能以 Mock 模式运行：
- 内置 3 城市（东京/巴黎/曼谷）× 3 档次（穷游/舒适/豪华）完整行程数据
- 航班/酒店/天气/汇率等工具返回模拟真实数据
- AI 助手能识别订票、订酒店、加行程、设提醒等操作意图并生成工具调用

## 技术亮点

### 1. 图编排 Agent（类 LangGraph）

```python
# 8 节点有向图，条件路由
load_context → guardrail_input → llm_call → parse_tools
    ↓ (blocked)                     ↓ (no tools)    ↓ (has actions)
  finalize                    format_reply    execute_tools → HITL 中断
                                     ↑              ↓ (confirmed)
                              guardrail_output ← format_reply
                                     ↓
                                  finalize
```

### 2. Human-in-the-Loop

操作类工具（book_flight, book_hotel, add_spot 等）触发中断：
- Agent 暂停执行，向用户展示操作详情
- 用户确认 → 执行操作并继续
- 用户取消 → 跳过操作，继续对话

### 3. MCP 兼容工具协议

每个工具声明 JSON Schema，统一注册和执行：

```python
ToolSchema(
    name="book_flight",
    description="预订航班",
    category="action",  # action = 需要 HITL 确认
    parameters={"type": "object", "properties": {
        "airline": {"type": "string", "description": "航班号"},
        ...
    }}
)
```

### 4. 双层安全护栏

- **输入护栏**：Prompt 注入检测 → 有害内容拦截 → PII 泄露警告
- **输出护栏**：有害内容过滤 → 格式验证
- 分级处理：critical 直接拦截，medium 出警告，low 放行

## 依赖清单

```
streamlit>=1.35
fastapi>=0.111
uvicorn>=0.30
httpx>=0.27
openai>=1.30
sqlalchemy>=2.0
pydantic>=2.7
python-dotenv>=1.0
```

## License

MIT
