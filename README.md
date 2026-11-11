

## 🛠 技术栈

| 方向 | 技术 |
|------|------|
| 后端 | Java · Spring Boot 3 · Go · Hertz · Node.js |
| AI / LLM | 图编排 Agent（ReAct）· RAG · MCP · SpringAI · LoRA 微调 |
| 推理 / 算子 | CUDA Kernel · vLLM · FP8 量化 · KV Cache 稀疏化 · 昇腾 AscendC |
| 数据 | MySQL · Redis · SQLite · Chroma |
| 前端 | Vue 3 · TypeScript · Vite · Element Plus |
| 其他 | Docker · Linux · Git |

## 📁 项目

### 🤖 AI / 大模型

| 项目 | 简介 | 亮点 |
|------|------|------|
| [TravelMate](./TravelMate) | AI 智能出行助手：图编排 ReAct Agent 全栈应用 | StateGraph 编排、HITL 确认、安全护栏、MCP 工具、RAG、多用户 |
| [MindGuard](./mindguard) | AI 心理监护助手：聊天 + 心理监控智能体 | SpringAI + ReAct + RAG + MCP + LoRA 微调 + JWT 鉴权 |
| [jev-demo](./jev-demo) | 电商客服意图路由沙盘 | 结构化决策模型做意图识别 / 情绪 / 紧急度分级与路由 |
| [sparsekv-engine](./sparsekv-engine) | 长文本推理 KV Cache 压缩加速系统 | 手写 Top-K 稀疏 Attention CUDA Kernel + FP8 量化，2~4 倍加速、显存降 50%+ |
| [cat2](./cat2) | 昇腾 CANN 自定义算子：SiLU 反向 | host 端 Tiling + AscendC kernel 实现（op_host / op_kernel） |
| [transformer](./transformer) | Transformer 从零实现（"Attention Is All You Need"） | 完整训练管线 + 消融实验，可复现 |

### ⚙️ 后端 / 分布式 / 全栈

| 项目 | 简介 | 亮点 |
|------|------|------|
| [task-dispatch](./task-dispatch) | 分布式脚本任务分发系统：5 节点调度 | 含设计文档 / 架构说明 / 故障演练手册 |
| [doudshengsheng](./doudshengsheng) | 兜省省：Redis 实战优惠平台 | Java / Go 双后端 + 红包雨 + AI 省钱助手 |
| [content-moderation](./content-moderation) | 投稿审核台：内部运营审核小工具 | Vue 3 + Node.js + SQLite，开箱即用 |
| [portfolio-simulator](./portfolio-simulator) | 投资组合分析与调仓模拟 | 组合概览、集中度超限识别、顺序调仓模拟 |
