# 兜省省 (DoudShengSheng)

模拟"抖省省"的本地优惠平台。**Redis 实战全栈项目**：Java / Go 双后端 + Vue3 前端 + 商户后台 + AI 助手。

基于黑马点评技术栈，新增**红包雨**与 **AI 省钱助手**两大亮点功能，并用 Java 和 Go 各实现了一套后端。


## 技术栈

| 端 | 技术 | 端口 |
|----|------|------|
| Java 后端 | SpringBoot 3.2 + MyBatis-Plus + Redis + Redisson + Lua + MySQL | 8081 |
| Go 后端 | Hertz + GORM + go-redis v9（复用同一套 MySQL / Lua 脚本，Redis 用 db1 隔离） | 8082 |
| 前端 | Vue3 + Vite + Vue Router + Element Plus + Axios | 5173 / 5174 |
| AI | DeepSeek（对话/Agent）+ Ollama `nomic-embed-text`（向量化） | — |

## 目录结构

```
doudshengsheng/
├── doudshengsheng/            # Java 后端(SpringBoot, 8081)
│   ├── src/main/java/com/dss/
│   │   ├── controller/        # 9 个 Controller(~44 个接口, 含商户后台 /admin)
│   │   ├── service/impl/      # 8 个业务实现(缓存三策略/秒杀/红包雨/社交/统计)
│   │   ├── ai/                # AI 模块: Agent(ReAct)+RAG+5个工具+流式
│   │   ├── aspect/            # @RateLimit 限流切面 + @AdminOnly 管理员鉴权切面
│   │   ├── interceptor/       # 双层拦截器(Token 续期 + 登录校验)
│   │   ├── utils/             # CacheClient(缓存三策略封装)、RedisIdWorker、Lua加载
│   │   ├── config/            # 布隆预热、启动预热、Jackson、Redisson 等
│   │   └── entity/ dto/ mapper/ exception/
│   ├── src/main/resources/
│   │   ├── lua/               # 3 个 Lua: seckill / redpacket_grab / sliding_window
│   │   └── application*.yml   # application-ai-key.yml 存 AI key(已 gitignore)
│   └── src/test/              # 30 个单测(红包拆分算法 27 + 全局ID并发 3)
├── doudshengsheng-go/         # Go 后端(Hertz, 8082), 与 Java 版功能对齐
│   ├── cmd/server/main.go     # 手动 wire 依赖
│   ├── internal/{handler,service,cache,middleware,ai,utils,model,config}
│   └── resources/lua/         # 复用 Java 版同一套 Lua
├── doudshengsheng-web/        # 前端(Vue3)
│   └── src/
│       ├── views/             # 12 个 C 端页面
│       ├── views/admin/       # 3 个商户后台页面
│       ├── api/index.js       # 43 个接口函数
│       └── router/ utils/ layout/
├── sql/
│   ├── schema.sql             # 建库建表(含 role/cover 列)
│   └── seed.sql               # 演示数据(可重复执行)
├── scripts/
│   ├── e2e_full.sh            # ★ 端到端全流程验证(10 大项)
│   ├── seckill_bench.sh       # 秒杀并发压测 + 超卖验证
│   ├── redpacket_bench.sh     # 红包雨并发压测 + 金额守恒验证
│   ├── grab_test.sh           # 多用户抢红包功能验证
│   └── gen_covers.py          # 生成商铺/分类占位图
├── INTERVIEW.md               # 面试讲解文档
└── README.md
```

## 快速启动

环境要求：Java 17+ / Maven 3.6+（Go 版需 Go 1.22+）/ Redis / MySQL / Node 18+

### 1. 数据库

```bash
mysql -uroot < sql/schema.sql               # 建库建表
mysql -uroot doudshengsheng < sql/seed.sql  # 演示数据(可重复执行)
```

> 已有旧库（缺 `role`/`cover` 列）补齐：
> `ALTER TABLE tb_user ADD COLUMN role TINYINT DEFAULT 0;`
> `ALTER TABLE tb_shop ADD COLUMN cover VARCHAR(255) DEFAULT '';`
>
> seed.sql 已把用户 1 设为管理员（role=1），可用其登录商户后台。

### 2. Java 后端（8081）

```bash
cd doudshengsheng
mvn clean package -DskipTests
java -jar target/doudshengsheng-1.0.0.jar
# 开发模式: mvn spring-boot:run
```

启动时自动完成：秒杀券库存预热、商铺 GEO 预热、布隆过滤器预热、无进行中红包时自动创建「每日红包雨」（50 元/10 个）。

### 3. Go 后端（8082，可选）

```bash
cd doudshengsheng-go
export DEEPSEEK_API_KEY=sk-xxx   # AI 功能需要,不用 AI 可跳过
go run ./cmd/server              # 复用同一套 MySQL,Redis 用 db1 与 Java 版隔离
```

### 4. 前端

```bash
cd doudshengsheng-web
npm install
npm run dev        # 5173 → 代理 Java 后端 8081
npm run dev:go     # 5174 → 代理 Go 后端 8082
```

登录：任意手机号，验证码打印在后端控制台日志（Go 版 `grep 验证码 /tmp/dss-go.log`）。

### 5. AI 功能配置

| 端 | 配置方式 |
|----|----------|
| Java | 新建 `doudshengsheng/src/main/resources/application-ai-key.yml`（已 gitignore）：`ai.deepseek.api-key: sk-xxx`；RAG 向量化需本地 ollama 运行 `nomic-embed-text` |
| Go | `export DEEPSEEK_API_KEY=sk-xxx`；ollama 同上 |

不配置 AI key 不影响其他功能，仅 AI 页面不可用。

## 前端页面

### C 端（12 个）

| 页面 | 功能 |
|------|------|
| 登录 | 验证码登录，Token 存 localStorage |
| 商铺列表/详情 | 分类切换；详情页可切换缓存策略（旁路/互斥锁/逻辑过期）对比耗时 |
| 附近商铺 | GEO 按经纬度 + 距离搜索 |
| 优惠券秒杀 | 单次秒杀 / 我的订单 |
| 红包雨 | 创建 / 抢（打字机金额动画）/ 排行榜（3 秒自动刷新） |
| 探店笔记 | 热门列表、点赞、点赞榜、发布（推送到粉丝 Feed） |
| 笔记详情 | 点赞状态同步、点赞人列表 |
| 个人主页 | 用户信息 + 其笔记列表 + 关注操作 |
| 关注/Feed | 关注、共同关注、Feed 流滚动分页 |
| 签到/UV | BitMap 签到日历、连续签到、HyperLogLog UV |
| AI 省钱助手 | Agent 多步规划（SSE 流式展示工具调用过程）+ RAG 语义问答 |

### 商户后台（3 个，`/admin`，仅 role=1 可访问）

| 页面 | 功能 |
|------|------|
| 数据概览 | 订单统计、秒杀券库存、红包雨场次 |
| 秒杀券管理 | 创建券（自动预热库存）、查库存、手动预热 |
| 红包雨数据 | 场次列表、实时进度（Redis meta）与 DB 落库对照 |

## 功能模块 → Redis 技术点

| 模块 | 接口 | Redis 技术点 |
|------|------|------------|
| 短信登录 | `POST /user/code` `POST /user/login` | Hash 存 Session、双层拦截器自动续期 Token |
| 商铺缓存 | `GET /shop/{id}?strategy=pass-through\|mutex\|logical` | 旁路缓存；穿透=布隆过滤器+空值缓存双层；击穿=互斥锁/逻辑过期双方案可切换；雪崩=随机 TTL |
| 附近商铺 | `GET /shop/of/near` | GEO + GeoSearch |
| 优惠券秒杀 | `POST /voucher/order/seckill/{id}` | 全局唯一 ID、Lua 原子预检+扣减、Stream 异步落库、Redisson 分布式锁兜底一人一单、**DB 乐观锁(`stock>0` CAS)兜底超卖**、唯一键兜底重复单 |
| **红包雨** | `POST /redpacket/create` `POST /redpacket/grab/{id}` `GET /redpacket/rank/{id}` | 二倍均值法、List 预分配、Lua 原子抢（幂等+取金额+记录+减余量）、滑动窗口限流、Stream 批量落库、到期未领退款 |
| 探店点赞 | `PUT /blog/like/{id}` | ZSet 一人一赞 + 排行榜 |
| 关注/Feed | `GET /follow/common/{id}` `GET /follow/feed` | Set 交集、ZSet 推模式滚动分页 |
| 签到 | `POST /stats/sign` | BitMap（一人一月 4 字节） |
| UV 统计 | `GET /stats/uv/count` | HyperLogLog（12KB，误差 0.81%） |
| 异步落库 | — | Stream 消费组（异常退避防死循环，批量 ACK） |
| AI 助手 | `POST /ai/chat` `POST /ai/rag` | Function Calling 5 个工具查业务数据，RAG 向量检索+关键词降级 |

九种数据结构全覆盖：String / Hash / List / Set / ZSet / GEO / BitMap / HyperLogLog / Stream。

### 秒杀防线层次

```
Lua 原子预检预扣(Redis, 第一道) → Stream 异步削峰
→ Redisson 分布式锁(一人一单兜底) → DB 乐观锁 stock>0 CAS(超卖兜底)
→ 唯一键 uk_user_voucher(重复单兜底)
```

## AI 省钱助手

- **Agent 模式**（`/ai/chat`，SSE 流式）：ReAct 循环（最多 6 步），LLM 自主调用工具：`search_shops` / `list_vouchers` / `list_redpackets` / `seckill_voucher` / `grab_redpacket`，前端实时展示每步思考与工具调用
- **RAG 模式**（`/ai/rag`）：商铺+笔记向量化入库（ollama），语义检索后由 LLM 生成回答；ollama 不可用时自动降级关键词检索
- Java 用 OkHttp + SseEmitter，Go 用 net/http + io.Pipe，同一套工具定义

## 工程化能力

| 能力 | 说明 |
|------|------|
| 全局异常处理 | `GlobalExceptionHandler` 统一异常 → 标准 Result |
| 参数校验 | DTO `@Valid` + `@Pattern`，Controller `@Validated` |
| 接口文档 | Java: springdoc Swagger `http://localhost:8081/swagger-ui.html`；Go: 自研文档页 `http://localhost:8082/swagger` |
| 权限 | `@AdminOnly` 注解 + AOP 切面（403），双层拦截器登录校验 |
| 限流 | 秒杀接口 `@RateLimit`（Redisson 令牌桶，IP 维度）；抢红包走 `sliding_window.lua`（用户级滑动窗口） |
| 单元测试 | 30 个 case：红包拆分算法不变量（总额守恒/每人≥1分/随机参数）27 个 + 全局 ID 并发唯一性 3 个 |
| Long 精度 | `JacksonConfig` 把 Long 序列化为 String，避免雪花 ID 超 JS 2^53 丢精度 |

## 验证与压测

```bash
# 1. ★ 端到端全流程验证(登录→缓存三策略→秒杀→红包雨→点赞→关注→签到→UV→后台→异常码)
bash scripts/e2e_full.sh

# 2. 秒杀并发 + 超卖验证(自动注册用户/重置库存/并发抢/校验 DB 无超卖)
bash scripts/seckill_bench.sh 500 200

# 3. 红包雨并发 + 金额守恒验证(300 用户抢 50 个 100 元)
bash scripts/redpacket_bench.sh 300 100 50

# 4. 商铺缓存读性能
ab -n 3000 -c 100 http://localhost:8081/shop/1
```

实测（本机 12 核，Redis/MySQL/后端同机）：

| 场景 | 结果 | 验证点 |
|------|------|--------|
| 商铺缓存读 | ~8000 QPS / 12ms，0 失败 | 缓存命中不碰 DB |
| 秒杀 500 抢 200 | 成功 200 / 失败 300（全部"库存不足"）/ **0 超卖**，DB 落库一致 | Lua 原子 + 乐观锁 |
| 红包雨 300 抢 50（100 元） | 50 人抢到、**金额守恒 10000 分**、0 超领 | 二倍均值法 + Lua 原子 |

压测曾发现并修复 Stream 消费者死循环拖垮服务的真实 bug（catch 后无退避 → 加 sleep 退避 + 中断退出）。

## 配置说明

`doudshengsheng/src/main/resources/application.yml` 关键项：

- `dss.redpacket.refund-delay-seconds`：未领红包退款延迟（演示 120s）
- `dss.redpacket.rate-limit-window-seconds` / `rate-limit-max`：抢红包滑动窗口限流参数
- `spring.profiles.include: ai-key`：AI 配置文件（gitignore，见上文模板）

Go 版：`config.yaml`（`DEEPSEEK_API_KEY` 环境变量注入 AI key，`config.local.yaml` 为本地覆盖示例，已 gitignore）。

