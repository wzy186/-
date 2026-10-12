# 投稿审核台（content-moderation）

内部运营审核用户投稿的小工具：看列表、勾选多条、批量通过 / 驳回，附操作日志与统计看板。值班同事开箱即用。

## 技术栈

- 前端：Vue 3 + TypeScript + Vite + Element Plus + Vue Router
- 后端：Node.js + TypeScript + Express
- 数据：本地 SQLite（使用 Node 内置 `node:sqlite`，**无原生编译依赖**，不需要安装任何数据库）

## 快速开始

要求：Node.js ≥ 22.13（推荐 24.x，开发环境为 v24.16）

```bash
cd content-moderation
npm install
npm run dev
```

- 前端开发服务器：http://localhost:5173 （`/api` 自动代理到后端 3000）
- 后端 API：http://localhost:3000/api/health
- 首次启动自动建库并写入演示数据（48 条投稿：30 条待审核 + 18 条已审历史，跨过去 7 天）

生产模式（后端托管前端构建产物，一个端口跑全部）：

```bash
npm run build
npm start        # http://localhost:3000
```

重置演示数据：

```bash
npm run seed     # 建议在服务停止时执行（服务运行中执行也可以，写锁冲突会自动等待重试）
```

运行测试：

```bash
npm test         # 服务端 API 集成测试（node:test，跑在独立临时数据库上，不碰演示数据）
```

## 功能

- **审核队列**：状态 Tab（待审核 / 已通过 / 已驳回 / 全部，带数量角标）、关键词搜索（标题 / 内容 / 作者）、按提交时间排序、分页
- **批量处理**：勾选多条（跨页保留选择）→ 批量通过 / 批量驳回（驳回必须填原因）
- **单条处理**：通过 / 驳回（预设原因 + 自定义补充）/ 重新审核（打回待审核队列）
- **详情弹窗**：完整内容、图片预览、审核记录
- **操作日志**：按操作人 / 动作筛选，记录谁在何时对哪条投稿做了什么、驳回原因
- **统计看板**：待审核积压、今日已处理、今日 / 累计通过率、近 7 日处理量图、状态分布、最近动态

## 操作人区分

不做账号体系：页面右上角选择「当前操作人」即可（值班同事名单预置在库里），选择存在浏览器 localStorage，每次操作随请求记录到日志。够用、零登录成本。

## 状态流转

```
pending（待审核） ──approve──▶ approved（已通过）
      │  ▲                      │  ▲
      │  └────────reopen────────┤  │
      └───reject────────▶ rejected（已驳回）┘
```

驳回必须带原因；已审核条目可「重新审核」打回队列，打回时清空原审核信息。

## 目录结构

```
server/                  Express + SQLite（node:sqlite）
  src/db.ts              建库建表
  src/seed.ts            演示数据（npm run seed 重置）
  src/app.ts             全部 API（与启动逻辑分离，便于测试）
  src/index.ts           启动入口 + 生产模式静态托管
  test/api.test.ts       API 集成测试（node:test）
web/                     Vue 3 + Element Plus
  src/views/ReviewQueue.vue     审核队列（列表/批量/详情/驳回弹窗）
  src/views/ReviewLogs.vue      操作日志
  src/views/StatsDashboard.vue  统计看板
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/operators` | 值班同事（操作人）列表 |
| GET | `/api/submissions?status=&q=&sort=&page=&pageSize=` | 投稿列表（筛选/搜索/分页/排序） |
| GET | `/api/submissions/:id` | 投稿详情 |
| POST | `/api/submissions/:id/review` | 单个审核：`{action: approve\|reject\|reopen, reason?, operator}` |
| POST | `/api/submissions/batch-review` | 批量审核：`{ids, action: approve\|reject, reason?, operator}` |
| GET | `/api/logs?operator=&action=&page=&pageSize=` | 操作日志 |
| GET | `/api/stats` | 统计看板数据 |
| GET | `/api/health` | 健康检查 |
