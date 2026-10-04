# 电商客服意图路由沙盘（Jev 版）

模拟电商客服对话，实时看 **Jev**（TypeSafe AI 的 System One 结构化决策模型）怎么判、怎么路由、怎么拦。

> 本仓库基于 [kiler398/jev-demo](https://github.com/kiler398/jev-demo) 部署，已接入真实 Jev 模型（`jev-1.13.0`）验证通过。

```
用户/客服消息 ──▶ Jev systemone ──▶ 归一化 ──▶ 决策表 ──▶ 队列 + 优先级 + SLA
                （choice/score/noul）            └──▶ 拦截器 ──▶ block / warn + 改写建议
```

## 功能一览

| 判定项 | 题型 | 说明 |
|---|---|---|
| 一级意图 | choice | 14 类：物流查询/异常、退款退货、换货补发、商品咨询、价格优惠、订单修改、支付问题、发票开票、售后维修、账号问题、投诉举报、活动规则、闲聊其他 |
| **情绪强度** | score | 5 档：平静 → 略有不满 → 明显不满 → 愤怒 → 暴怒，返回分值+概率，看板实时画情绪曲线 |
| 紧急度 | score | 4 档：不急 → 一般 → 较急 → 非常急 |
| 该转人工 | noul | 返回概率（≥60% 触发转人工路由） |
| 用户侧风险 | choice | 监管投诉 / 媒体曝光 / 法律诉讼 / 人身攻击 / 疑似薅羊毛 |
| 危险话术 | choice | 10 类，block 级直接拦截，warn 级给改写建议 |
| 服务质量 | score | 5 档 + 是否有共情 + 是否给可执行方案 |
| 会话级分析 | 混合 | CSAT 预测、一次解决率、流失风险、归档标签（手动触发） |

8 条路由规则按优先级从上到下匹配：客诉专席 P0 → 风控核查 → 客诉专席 P1 → 高优先专线 → 物流专线 → 按意图自助。

## 快速开始

```bash
npm install
npm run dev          # http://127.0.0.1:5173
```

**不配 API Key 也能跑** —— 自动走内置 Mock 规则引擎（12 个内置剧本一键回放）。

### 接入真实 Jev 模型

1. 到 [console.typesafe.ai](https://console.typesafe.ai) 注册并创建 API Key
   （也可以用 [openrouter.ai](https://openrouter.ai) 的 key，`sk-or-v1-` 开头会自动走 OpenRouter 端点）
2. 写入 `.env`（项目根目录，参考 `.env.example`）：

```bash
TYPESAFE_API_KEY=你的key
JEV_MODEL=jev-latest
```

3. 重启 `npm run dev`，右上角引擎标识从「Mock」变为「Jev jev-latest」即接入成功

> 注意：`.env` 里的 `JEV_ENDPOINT` 留空即可，代码会按 key 前缀自动选择官方端点或 OpenRouter。TypeSafe 官方 key 走 `https://api.typesafe.ai/v1/systemone`。

### 生产模式

```bash
npm run build && npm start   # http://127.0.0.1:5180
```

其它命令：`npm test`（34 个单测）、`npm run typecheck`。

## 为什么用 Jev 而不是让大模型输出 JSON

这类场景要的是**有边界的判断**，不是生成文本：

- `choice` 从给定选项里选一个，返回**每个选项的概率** —— 意图路由、风险分类
- `score` 在有序档位上打分，返回分值（如 2.7）+ 每档概率 —— 情绪强度、服务质量、CSAT
- `noul` 是/否，返回概率 —— 该不该转人工、有没有共情

好处：输出**永远合法**，不用写 JSON 修复逻辑；带**概率和置信度**，阈值可调不用改提示词；`criteria` 里能给每个选项写判定口径，等于把业务标准写进接口。一次请求可问多个问题共享同一段上下文，input tokens 只算一次。

接口：`POST https://api.typesafe.ai/v1/systemone`，模型 `jev-latest`，计费 $0.042 / 1M input tokens（output 暂不计费）。一通 10 轮会话不到 0.0002 美元。

## 改什么在哪改

全是数据驱动，改 JSON 不用动代码，改完刷新页面就生效（后端每次请求重读）：

| 想改什么 | 改哪个文件 |
|---|---|
| 加/减判定项、改判定口径、改情绪档位 | `shared/policies.json` |
| 路由规则、队列、SLA、拦截级别与改写话术 | `shared/routing.json` |
| 演示剧本 | `shared/scripts.json` |
| Mock 的关键词表（只影响没配 key 时） | `server/mock.mjs` |

### 情绪档位示例（policies.json）

```json
{
  "id": "emotion",
  "type": "score",
  "label": "情绪强度",
  "instructions": "这位用户此刻的负面情绪有多强？看措辞、标点、重复和攻击性，不看事情本身严不严重。",
  "criteria": [
    "平静：正常陈述或询问，没有情绪词",
    "略有不满：有点着急或抱怨，但仍然客气",
    "明显不满：反复催促、质疑、带明显责备语气",
    "愤怒：指责、讽刺、感叹号连用、要求给说法",
    "暴怒：辱骂、威胁曝光或投诉、扬言起诉或找媒体"
  ]
}
```

### 路由条件写法（routing.json）

```json
{ "q": "intent",  "in": ["投诉举报"] }     // choice：命中其中之一
{ "q": "emotion", "gte": 2 }              // score：档位下标 >= 2
{ "q": "escalate","gte": 0.6 }            // noul：概率 >= 0.6
{ "any": [ ... ] }                        // 或
```

`when` 内是**与**，`any` 内是**或**。`queue: "byIntent"` 表示按意图查 `intentQueue` 表。

## 工程结构

```
shared/           判定策略、路由决策表、剧本 —— 前后端共用，改这里就改行为
server/
  api.mjs         API 层 + 上下文拼装 + 结果归一化（dev/prod 共用）
  jev.mjs         Jev 调用：问题切批并发、429/529 指数退避、用量统计
  mock.mjs        无 key 时的规则引擎，输出结构与 Jev 一致
  index.mjs       生产模式静态服务
src/
  lib/decide.ts   路由决策与拦截判定（纯函数，单测覆盖）
  lib/mask.ts     看板脱敏 —— 拦下来的手机号不能在看板上再泄露一次
  components/     对话面板、消息气泡、实时看板
test/             34 个单测：条件匹配、8 条路由规则、拦截分级、12 个剧本回归、脱敏
```

## 实测数据（jev-1.13.0）

输入：`我的退款都三天了还没到账，你们到底会不会处理？！再不退我就去12315投诉！`

| 判定项 | 结果 | 置信度 |
|---|---|---|
| 意图 | 退款退货 | 99% |
| 情绪 | 暴怒（3.96，暴怒档 96%） | 97% |
| 紧急度 | 一般（1.34） | 65% |
| 转人工 | 是（87%） | — |
| 用户风险 | 监管投诉（识别出 12315） | 100% |

单条消息判定耗时约 660ms，成本约 $0.00007。

## 已知边界

- Mock 是关键词规则，只用于演示和回归测试，**别拿它的准确率当模型的准确率**
- 每条消息带最近 8 轮上下文，长会话不会无限膨胀
- 会话级分析是手动触发的，真上线建议每 N 轮自动跑或会话结束时跑
- 拦截只作用在沙盘输入框，真接客服工作台要挂在发送按钮的前置钩子上
- ⚠️ API Key 只存 `.env`（服务端），**不要提交到仓库**，`.gitignore` 应包含 `.env`


- 模型与 API：[TypeSafe AI](https://typesafe.ai)（Jev / System One）
