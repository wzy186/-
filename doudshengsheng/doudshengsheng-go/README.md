# 兜省省 Go 版 (DoudShengSheng-Go)

兜省省的 **Go 语言版本**,功能与 Java 版完全对齐。技术栈对标字节跳动(Hertz + Kitex 生态),适合面字节 Go 岗。

## 技术栈

| 类别 | 技术 | 对应 Java 版 |
|------|------|------------|
| HTTP 框架 | **Hertz**(字节自研) | SpringBoot |
| Redis | go-redis v9 | Spring Data Redis + Redisson |
| ORM | GORM | MyBatis-Plus |
| 数据库 | MySQL(复用 Java 版同一套表) | 同 |
| Lua | go-redis Eval(脚本复用,不改) | RedisTemplate execute |
| 参数校验 | 手写 | @Valid |
| API 文档 | 自研轻量 Swagger | springdoc-openapi |
| LLM | net/http 调 DeepSeek | OkHttp |
| AI 流式 | io.Pipe + SSE | SseEmitter |

## 目录结构

```
doudshengsheng-go/
├── cmd/server/main.go         ← 启动入口(手动 wire 依赖)
├── config.yaml                ← 配置(复用 MySQL,Redis 用 db1 隔离)
├── resources/lua/             ← 3 个 Lua(从 Java 版复制,不改)
└── internal/
    ├── ai/                    ← AI 模块(llm/embedding/rag/tool/agent)
    ├── cache/                 ← 缓存客户端(泛型三策略+布隆)
    ├── config/                ← 配置加载
    ├── handler/               ← 8 个 handler(user/shop/voucher/redpacket/social/ai/doc)
    ├── middleware/            ← CORS/异常/Token/登录/管理员
    ├── model/                 ← GORM 实体
    ├── service/               ← 业务逻辑
    └── utils/                 ← DB/Redis/ID/常量/用户上下文
```

## 功能模块(对齐 Java 版,8 种 Redis 数据结构全覆盖)

| 模块 | Redis 结构 | 亮点 |
|------|----------|------|
| 短信登录 | Hash | 双中间件(刷新Token+校验登录),context 传用户 |
| 商铺缓存 | String | 三策略(旁路/互斥锁/逻辑过期)+ 布隆+空值防穿透 + 随机TTL防雪崩 |
| 附近商铺 | GEO | GeoSearch |
| 优惠券秒杀 | String+Set+Stream | Lua 原子扣库存 + 分布式锁 + goroutine 异步落库 |
| **红包雨** | List+Hash+Stream | 二倍均值法 + Lua 原子抢 + 滑动窗口限流 + goroutine 落库 + 延迟退款 |
| 点赞 | ZSet | 一人一赞 + 排行榜 |
| 关注/Feed | Set+ZSet | 交集共同关注 + 推模式滚动分页 |
| 签到 | BitMap | 连续签到 + 本月记录 |
| UV | HyperLogLog | 去重计数 |
| **AI 助手** | 内存向量 | RAG 语义检索 + Agent ReAct + Function Calling + SSE 流式 |

## 启动

```bash
cd doudshengsheng-go

# 1. 配置(改 config.yaml,复用 Java 版的 MySQL 库)
#    Redis 用 db1,和 Java 版(db0)隔离,避免 key 冲突

# 2. 启动 Go 后端(8082)
export GOPROXY=https://goproxy.cn,direct
go run ./cmd/server

# 3. 启动 Go 版前端(5174,代理到 8082)
cd ../doudshengsheng-web
npm run dev:go
```

浏览器打开 http://localhost:5174 ,手机号 `13800000001`,验证码看 Go 后端日志(`grep 验证码 /tmp/dss-go.log`)。

## API 文档

启动后访问 http://localhost:8082/swagger (可搜索的接口列表页),或 http://localhost:8082/swagger/apis (JSON)。

## 对比 Java 版的 Go 优势(面试讲点)

1. **goroutine 消费 Stream**:秒杀订单/红包记录的异步落库用 `go consumeLoop()` 一行启动,比 Java 线程池 + while 循环简洁
2. **泛型缓存客户端**:`QueryWithPassThrough[T any]` 真泛型,Java 擦除做不到
3. **SSE 流式**:`io.Pipe` + `bufio.Scanner` 读 SSE,比 Java SseEmitter 直观
4. **context 传用户**:Go 用 `context.Context` 传递用户态,比 Java ThreadLocal 更显式、无内存泄漏风险
5. **手动 wire 依赖**:依赖关系在 main 里一目了然,比 Spring @Autowired 隐式注入好排查
6. **RAG 并发安全**:`sync.RWMutex` 保护向量 map,读写分离

## 压测

复用 Java 版脚本(改端口 8081 → 8082):
```bash
# 秒杀(改脚本里 BASE=http://localhost:8082)
bash scripts/seckill_bench.sh 500 200
# 红包雨
bash scripts/redpacket_bench.sh 300 100 50
```

## 两版共存

- Java 版:后端 8081,前端 5173(`npm run dev`)
- Go 版:后端 8082,前端 5174(`npm run dev:go`)
- 同一套 MySQL,Redis 用不同 db 隔离

面试可讲"我用 Java 和 Go 都实现了同一套系统,对比了并发模型/依赖注入/流式处理"。
