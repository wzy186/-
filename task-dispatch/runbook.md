# 启动说明（Runbook）

## 环境要求

- Node.js ≥ 22.13（推荐 24.x；使用内置 `node:sqlite`，无任何原生编译依赖）
- 无需安装数据库、无需 Python（执行环节为 mock；如需真实执行 Python，机器上有 python3 即可，见下文）

## 目录结构

```
task-dispatch/
├── design.md          # 详细设计文档
├── architecture.md    # 架构与技术设计说明
├── runbook.md         # 本文档
├── README.md          # 总览
├── package.json       # 一个依赖 express 的单一项目，两种入口
├── src/
│   ├── coordinator.ts # 调度中心（唯一状态写者，SQLite）
│   ├── worker.ts      # 执行节点（每进程一个节点，可起 N 个）
│   ├── executor.ts    # mock 执行器（接口抽象，可切真实 python3）
│   └── lib/           # 状态机、领取 SQL、回收器等共用逻辑
├── test/              # 领取原子性、超时回收、幂等等测试
└── scripts/
    ├── start-all.sh   # 一键启动：1 个 Coordinator + 5 个 Worker
    └── demo.sh        # 演示脚本（批量提交/故障演练）
```

## 快速启动（一键）

```bash
cd task-dispatch
npm install
./scripts/start-all.sh
```

脚本依次拉起：1 个 Coordinator（端口 3000）+ 5 个 Worker（worker-1 ~ worker-5），日志写入 `logs/`。看到 `✅ 调度中心已启动` 与 5 行 `✅ worker-N 已注册` 即就绪。

## 手动启动

```bash
# 终端 1：调度中心
npm run coordinator        # 默认端口 3000，首次启动自动建库

# 终端 2~6：5 个执行节点（每个独立进程 = 一个节点）
WORKER_ID=worker-1 npm run worker
WORKER_ID=worker-2 npm run worker
WORKER_ID=worker-3 npm run worker
WORKER_ID=worker-4 npm run worker
WORKER_ID=worker-5 npm run worker
```

Worker 启动后自动注册到调度中心，并开始轮询领取任务（无需先起后起顺序约定；Coordinator 后启动也行，Worker 会持续重连）。

## 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PORT` | 3000 | Coordinator 端口 |
| `COORDINATOR_URL` | `http://127.0.0.1:3000` | Worker 连接的调度中心地址 |
| `WORKER_ID` | 自动生成 | 节点标识（手动启动时建议显式指定） |
| `WORKER_CONCURRENCY` | 3 | 单个节点内的并发任务数 |
| `HEARTBEAT_INTERVAL_MS` | 2000 | 心跳间隔 |
| `SWEEP_INTERVAL_MS` | 5000 | 超时回收扫描间隔 |
| `LEASE_TIMEOUT_MS` | 10000 | 租约超时（心跳过期阈值） |
| `MAX_RETRY` | 3 | 任务最大重试次数 |
| `MOCK_DURATION_RANGE` | `200-1500` | mock 执行耗时范围（毫秒） |
| `EXECUTOR` | `mock` | `mock` = 模拟执行；`real` = 调 python3 真执行（需本机有 python3） |
| `INTERNAL_TOKEN` | 空 | 内部接口鉴权 token（真实部署时设置） |

## 提交与查询示例

```bash
# 提交任务
curl -X POST http://127.0.0.1:3000/api/tasks \
  -H 'Content-Type: application/json' \
  -d '{"script": "print(sum(range(100)))"}'
# → {"taskId":"tk_ab12cd","status":"queued"}

# 查询状态与结果
curl http://127.0.0.1:3000/api/tasks/tk_ab12cd
# → {"taskId":"tk_ab12cd","status":"succeeded","result":"4950",...}

# 只取结果（未完成时返回 409）
curl http://127.0.0.1:3000/api/tasks/tk_ab12cd/result

# 批量提交 30 个任务
for i in $(seq 1 30); do
  curl -s -X POST http://127.0.0.1:3000/api/tasks \
    -H 'Content-Type: application/json' \
    -d "{\"script\": \"print($i * 2)\"}" &
done; wait
```

## 停止与重启

```bash
./scripts/stop-all.sh          # 按 PID 文件停止全部进程
./scripts/start-all.sh         # 再次启动；任务状态在 SQLite 中，重启不丢
```

单独重启某个 Worker：kill 对应进程后按"手动启动"方式重新拉起即可——其余节点不受影响。

## 故障演练（验证稳定性）

1. 提交一批任务，待部分进入 running；
2. `kill` 掉某个 Worker 进程（或直接关掉它的终端）；
3. 观察：≤10s 内其持有的 running 任务回到 queued 并改由其他节点执行，最终全部 succeeded；
4. 恢复该 Worker：重新启动即自动注册归队。

## 运行测试

```bash
npm test
```

覆盖：领取原子性（并发抢单无重复）、状态机合法/非法迁移、超时回收与重试上限、幂等回写、重启恢复、接口契约。
