# 分布式脚本任务分发系统 — 详细设计文档

## 1. 目标

业务方通过**统一接口**提交 Python 脚本任务并查询状态与结果，感知上只是调用一个"普通计算接口"，无需关心任务被分到哪台节点、如何分配。系统要求：5 个执行节点同时工作、同一任务只能被一个节点执行一次、任务完成后结果可被正确查询、大量任务涌入时稳定运行。

在本次开发中，5 个节点以同一台机器上的 5 个独立进程模拟；但设计上按真实分布式对待：**节点之间只通过 HTTP 网络通信协调，不依赖任何共享内存、全局变量**；节点内部可自由使用异步并发。

## 2. 总体架构

```
                         ┌────────────────────────────────────────────┐
  业务方 ──HTTP──▶       │  调度中心 Coordinator（唯一入口/唯一状态写者）│
  POST /api/tasks        │      Express + node:sqlite（同步驱动）       │
  GET  /api/tasks/:id    └───────────────┬────────────────────────────┘
                                         │ SQLite（tasks / workers 表）
                                         │ HTTP 内部接口（仅网络通信）
      ┌──────────────┬──────────────┬────┴─────┬──────────────┬──────────────┐
      ▼              ▼              ▼          ▼              ▼              ▼
  worker-1       worker-2       worker-3    worker-4       worker-5
  (独立进程)     (独立进程)     (独立进程)  (独立进程)     (独立进程)
  并发池×3       并发池×3       并发池×3    并发池×3       并发池×3
      └── 轮询领取 → mock 执行 → 回写结果 → 心跳（2s）──┘
```

| 组件 | 形态 | 职责 |
|---|---|---|
| Coordinator | 1 个进程 | 任务提交入队、原子发放、结果落库、心跳登记、超时回收、查询服务 |
| Worker ×5 | 5 个独立进程 | 领取任务 → 模拟执行 → 回写结果；无任何本地持久状态 |
| SQLite | 文件数据库 | 任务与节点状态的唯一事实源，由 Coordinator 独占写 |

## 3. 任务状态机

```
                提交                        执行中                 成功
  (无) ──────▶ queued ──claim──▶ running ──complete(ok)──▶ succeeded
                 ▲                  │  │
                 │       超时且未达上限│ └──complete(fail)/脚本校验失败/重试耗尽──▶ failed
                 └──requeue(retry+1)─┘
```

| 迁移 | 触发者 | 条件 | 动作 |
|---|---|---|---|
| → queued | 业务方 | 提交脚本 | 生成 taskId，入队，立即返回 |
| queued → running | Worker | 原子领取成功 | 记录 worker_id、claimed_at |
| running → succeeded | Worker | 回写 result | 记录结果、finished_at |
| running → failed | Worker | 执行异常/脚本非法 | 记录 error |
| running → queued | Coordinator | 心跳过期（>10s）且 retry < max_retry | retry+1，清 worker 归属 |
| running → failed | Coordinator | 心跳过期且 retry ≥ max_retry | 记录"重试耗尽" |
| running → queued | Coordinator | 调度中心重启时 | 重启扫描兜底（见 7.4） |

## 4. 数据模型（SQLite DDL）

```sql
CREATE TABLE tasks (
  id           TEXT PRIMARY KEY,          -- taskId：随机短 ID，如 tk_ab12cd
  script       TEXT NOT NULL,             -- 要执行的 Python 脚本原文
  status       TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
  worker_id    TEXT,                      -- 当前持有者（领取时写入）
  claimed_at   TEXT,                      -- 领取时间
  heartbeat_at TEXT,                      -- 最近心跳时间
  retry_count  INTEGER NOT NULL DEFAULT 0,
  max_retry    INTEGER NOT NULL DEFAULT 3,
  result       TEXT,                      -- stdout 或 mock 结果
  error        TEXT,                      -- 失败原因
  created_at   TEXT NOT NULL,
  started_at   TEXT,
  finished_at  TEXT
);
CREATE INDEX idx_tasks_status ON tasks(status, created_at);

CREATE TABLE workers (
  id         TEXT PRIMARY KEY,            -- worker-1 … worker-5
  started_at TEXT,
  last_heartbeat TEXT
);
```

- **id 自生成**（Coordinator 侧 `crypto.randomBytes` 短 ID），不暴露自增序号。
- `script` 原样保存，不解析不执行（执行在 Worker 侧，mock 或真实执行均可）。

## 5. 接口设计

### 5.1 面向业务方（公开接口）

**提交任务**

```
POST /api/tasks
{ "script": "print(sum(range(100)))" }
→ 202 { "taskId": "tk_ab12cd", "status": "queued" }
错误：400 脚本为空/超长（>64KB）
```

**查询任务状态**

```
GET /api/tasks/:id
→ 200 {
  "taskId": "tk_ab12cd", "status": "succeeded",
  "result": "4950", "error": null,
  "workerId": "worker-3", "retryCount": 0,
  "createdAt": "...", "startedAt": "...", "finishedAt": "..."
}
错误：404 任务不存在
```

**查询结果**（语义更明确的便捷接口）

```
GET /api/tasks/:id/result
→ 200 { "taskId": "...", "status": "succeeded", "result": "4950" }
错误：404 不存在；409 尚未完成（status 为 queued/running 时）
```

### 5.2 内部接口（Worker ↔ Coordinator，仅网络通信）

| 接口 | 说明 |
|---|---|
| `POST /api/internal/claim` `{workerId}` | 原子领取一个 queued 任务；队列空返回 `{task: null}` |
| `POST /api/internal/complete` `{workerId, taskId, ok, result?, error?}` | 回写执行结果；校验归属与状态，不合法返回 409 |
| `POST /api/internal/heartbeat` `{workerId}` | 更新 workers.last_heartbeat 及该 worker 名下 running 任务的 heartbeat_at |
| `POST /api/internal/register` `{workerId}` | 节点启动注册（幂等 upsert） |

安全说明：同机模拟时内部接口仅绑定回环地址（或共享一个环境变量 token）；文档注明真实部署须加鉴权，鉴权不影响调度核心逻辑。

## 6. 核心机制与时序

### 6.1 恰好一次领取（原子领取协议）

领取动作由 Coordinator 以**单条原子 SQL** 完成：

```sql
UPDATE tasks
SET status='running', worker_id=:workerId, claimed_at=:now,
    heartbeat_at=:now, started_at=:now
WHERE id = (SELECT id FROM tasks
            WHERE status='queued'
            ORDER BY created_at, id LIMIT 1)
RETURNING id, script, retry_count;
```

**为什么不可能重复执行：**

1. Coordinator 是任务状态的**唯一写者**，且 node:sqlite 为同步 API——单进程内所有请求串行处理，不存在并发写交错；
2. 即使未来 Coordinator 变多写者，SQLite 的写事务互斥锁也保证两条相同 SQL 串行提交：后执行的那条 `WHERE status='queued'` 已不命中，`RETURNING` 为空；
3. Worker 拿到 RETURNING 的行才算领到任务；**领到即变更状态**，不存在"先领后变"的窗口期。

时序：

```
Worker                          Coordinator
  │ POST /claim {workerId}          │
  │───────────────────────────────▶ │ 原子 UPDATE ... RETURNING
  │  ◀── {task: {id, script}}       │   （或 {task: null} 队列空）
  │ [mock 执行，耗时可配]            │
  │ POST /complete {taskId, ok,..}  │
  │───────────────────────────────▶ │ 校验 running+归属 → 落库
  │  ◀── 200 / 409                   │
```

### 6.2 结果回写与幂等

回写必须同时满足：`status='running' AND worker_id=:workerId`，否则返回 409 并**不修改任何数据**。这保证：

- Worker 执行超时被回收后，其**迟到的回写被拒绝**（任务已重新入队或已由他人执行）——结果不会被错误覆盖；
- Worker 因网络重试重复 POST 同一结果：第一次成功后任务已非 running，第二次直接 409——结果只落库一次。

### 6.3 心跳与超时回收（租约思想）

| 参数 | 默认值 | 含义 |
|---|---|---|
| 心跳间隔 | 2s | 每个 Worker 定时上报 |
| 回收扫描间隔 | 5s | Coordinator 定时任务 |
| 租约超时 | 10s | running 任务心跳超过该时长视为节点失联 |

```sql
-- 回收：失联且未达重试上限 → 重新入队
UPDATE tasks SET status='queued', worker_id=NULL, claimed_at=NULL,
                heartbeat_at=NULL, retry_count=retry_count+1
WHERE status='running'
  AND heartbeat_at < :now - 10s
  AND retry_count < max_retry;

-- 重试耗尽 → 失败
UPDATE tasks SET status='failed', error='重试耗尽：执行节点失联'
WHERE status='running' AND heartbeat_at < :now - 10s
  AND retry_count >= max_retry;
```

心跳带来的额外保证：**节点崩溃或网络分区时任务不丢失**——最迟 10s 后重新入队，由存活节点继续执行；重试 3 次仍失败则明确置 failed，业务方可感知。

### 6.4 重启恢复

- **Coordinator 重启**：状态全部持久化在 SQLite。启动时扫描 `running` 任务统一置回 `queued`（retry+1）——重启本身视为一次"节点失联"。
- **Worker 重启**：无本地状态，重启后直接重新注册、继续轮询领取即可。

## 7. 故障场景推演

| 场景 | 系统行为 | 业务方感知 |
|---|---|---|
| 某 Worker 进程被 kill（执行中） | ≤10s 心跳过期 → 任务重新入队 → 其他节点执行 | 状态短暂 running，随后正常完成 |
| Worker 网络分区（进程活着但不通） | 同上（心跳停止） | 同上 |
| Worker 分区恢复后迟到回写 | 回写校验失败 409，结果被忽略 | 以实际执行者的结果为准 |
| Coordinator 崩溃 | 所有状态已持久化；重启后 running → queued | 重启后继续处理，无任务丢失 |
| 大量任务瞬间涌入 | 提交 O(1) 入队立即返回；5 节点 ×3 并发持续消化，队列自然缓冲 | 无拒绝，只影响排队时长 |
| 脚本为空/非法 | 空脚本 400；非法脚本由 Worker 判定执行失败并回写 error | 状态 failed + 原因 |
| 队列为空 | Worker 轮询拿到 `{task: null}`，等待下一轮 | — |

## 8. 正确性论证（恰好一次执行）

- **最多执行一次**：状态机前置条件保证——只有 `queued` 可被领取（原子 SQL 的 WHERE），只有 `running` 且归属一致的 Worker 能回写结果；领取与回写都是"校验 + 变更"合一的原子语句。
- **至少执行一次（直到成功或明确失败）**：任务进入 running 后，要么被回写终态，要么租约过期被回收；回收有重试上限，上限内总能再次领取。终态（succeeded/failed）一旦写入不可再迁移——不存在"卡死在中间态"。
- **结果唯一**：回写幂等校验使同一任务的结果最多落库一次，且迟到/重复回写被拒，不存在两个节点先后写入同一任务结果。

## 9. 测试与演示方案

单元/集成测试（Coordinator 内，node:test）：

1. **领取原子性**：5 个并发客户端同时 claim，队列 N 个任务 → 每个任务恰好被领取一次、无重复无遗漏；
2. **状态机合法迁移**：覆盖 8 种迁移 + 非法迁移被拒（如 succeeded 再回写 → 409）；
3. **超时回收**：注入过期 heartbeat → 扫描后任务重新入队且 retry+1；重试耗尽 → failed；
4. **幂等回写**：重复 complete → 第二次 409，结果不变；
5. **重启恢复**：模拟 running 残留，重启逻辑后全部回到 queued；
6. **接口契约**：提交/查询/结果的正常与错误码路径。

演示脚本（demo.sh，多终端窗口）：

- 场景 A：一次提交 30 个任务，观察 5 节点分摊消化、状态流转与结果正确；
- 场景 B：执行中 kill 一个 worker，观察其持有任务 10s 内被其他节点接手；
- 场景 C：提交非法脚本，观察 failed 与错误原因；
- 场景 D：空队列时节点空转、心跳正常。

## 10. 扩展性（真实分布式演进路径）

本设计已把"节点间协调"收敛为纯网络协议（claim/complete/heartbeat），单机模拟与真实分布式的差别只在部署形态：

| 当前（本作业） | 真实分布式演进 |
|---|---|
| SQLite 文件库，Coordinator 单点 | 队列/存储换 Redis+DB，调度核心逻辑（原子领取、租约回收、幂等）不变 |
| Coordinator 单进程 | 主备 + 选主（租约锁），或领取语句换成 Redis `BLMOVE` 原子原语 |
| Worker 轮询领取（拉模型） | 可换推送（Coordinator 主动下发），需加节点注册发现与健康检查 |
| 5 节点固定 | Worker 启动注册制，动态扩缩容 |
| 结果存任务行 | 大结果走对象存储，任务行只存引用 |
| 回环 HTTP 明文 | 内部接口加 token/mTLS 鉴权 |
