# MindGuard AI 心理监护助手 — 技术文档

## 一、项目概述

MindGuard AI 是一个面向高校学生的心理健康监护智能助手，基于 **Agent + RAG + MCP** 三层架构构建。系统不仅能进行共情对话，还能自主检索知识库、检测风险等级、自动发送告警邮件、写入心理记录、预约辅导员——是一个具备「感知-推理-行动」闭环的 Agentic RAG 系统。

**核心技术栈**：

| 层级 | 技术 |
|------|------|
| 框架 | Spring Boot 3.3.6 + JDK 17 |
| AI 框架 | Spring AI 1.0.0-M5 |
| 大模型 | Ollama (qwen2.5:7b) / OpenAI (gpt-4o) |
| 向量存储 | SimpleVectorStore（本地内存） |
| 数据库 | MySQL 8.x + Spring Data JPA |
| 认证 | Spring Security + JWT (jjwt 0.12.5) |
| 前端 | 单页 HTML + 原生 JS（暗色主题 AI 对话界面） |
| 外部服务 | JavaMail（邮件）、Apache POI（Excel） |

**项目结构**（38 个源文件，约 2000 行 Java 代码）：

```
mindguard/
├── pom.xml
├── src/main/
│   ├── java/com/mindguard/
│   │   ├── MindGuardApplication.java        # 启动入口
│   │   ├── agent/
│   │   │   └── MindGuardAgent.java           # ReAct Agent 核心
│   │   ├── config/
│   │   │   ├── ModelStrategyConfig.java      # 模型策略配置
│   │   │   └── VectorStoreConfig.java        # 向量存储配置
│   │   ├── controller/
│   │   │   ├── AuthController.java           # 认证接口
│   │   │   ├── ChatController.java           # 聊天+记录查询接口
│   │   │   └── AdminController.java          # 管理员知识库接口
│   │   ├── exception/
│   │   │   └── GlobalExceptionHandler.java   # 全局异常处理
│   │   ├── memory/
│   │   │   ├── ShortTermMemory.java          # 短期记忆（滑动窗口）
│   │   │   ├── LongTermMemory.java           # 长期记忆（向量检索）
│   │   │   └── MemoryService.java            # 记忆协调器
│   │   ├── model/
│   │   │   ├── User.java                     # 用户实体
│   │   │   ├── ChatSession.java              # 会话实体
│   │   │   ├── ChatMessage.java              # 消息实体
│   │   │   ├── PsychologicalRecord.java      # 心理记录实体
│   │   │   ├── Role.java                     # 角色（ADMIN/USER）
│   │   │   ├── RiskLevel.java                # 风险等级枚举
│   │   │   └── EmotionLabel.java             # 情绪标签枚举
│   │   ├── repository/
│   │   │   ├── UserRepository.java
│   │   │   ├── ChatSessionRepository.java
│   │   │   ├── ChatMessageRepository.java
│   │   │   └── PsychologicalRecordRepository.java
│   │   ├── security/
│   │   │   ├── SecurityConfig.java           # 安全配置
│   │   │   ├── JwtAuthFilter.java            # JWT 过滤器
│   │   │   └── JwtUtil.java                  # JWT 工具类
│   │   ├── service/
│   │   │   ├── AuthService.java              # 认证服务
│   │   │   ├── ChatService.java              # 聊天服务
│   │   │   ├── DocumentService.java          # 知识库服务
│   │   │   └── McpClientService.java         # MCP 外部服务封装
│   │   └── tools/
│   │       ├── RagSearchTool.java            # RAG 检索工具
│   │       ├── QueryHistoryTool.java         # 历史查询工具
│   │       ├── SendEmailTool.java            # 邮件告警工具
│   │       ├── WriteExcelTool.java           # Excel 写入工具
│   │       └── BookCounselorTool.java        # 辅导员预约工具
│   └── resources/
│       ├── application.yml                   # 应用配置
│       ├── schema.sql                        # 数据库建表脚本
│       └── static/index.html                 # 前端单页应用
└── src/test/java/                            # 单元测试
```

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────┐
│                    前端 (index.html)                  │
│   暗色主题 AI 对话界面 · 登录注册 · 记录面板           │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / JWT
                       ▼
┌─────────────────────────────────────────────────────┐
│              Controller 层 (API 接口)                 │
│  AuthController  ChatController  AdminController     │
└──────┬──────────────┬──────────────┬────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│AuthService│  │ ChatService  │  │DocumentService│
│ 认证注册  │  │ 聊天+记忆    │  │ 知识库RAG     │
└──────────┘  └──────┬───────┘  └──────────────┘
                     │
                     ▼
           ┌───────────────────┐
           │  MindGuardAgent   │  ← ReAct 推理引擎
           │  (Agent 核心)     │
           └──┬────┬────┬────┬┘
              │    │    │    │
     ┌────────┘    │    │    └────────┐
     ▼             ▼    ▼             ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│RagSearch│ │Query   │ │SendMail│ │WriteExcel│
│  Tool   │ │History │ │  Tool  │ │   Tool   │
│(RAG检索)│ │Tool    │ │(告警)  │ │(记录写入)│
└────┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘
     │          │          │           │
     ▼          ▼          ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│Vector  │ │Vector  │ │JavaMail│ │Apache POI│
│Store   │ │Store   │ │邮件服务│ │Excel服务 │
│(知识库)│ │(长期记忆)│ └────────┘ └──────────┘
└────────┘ └────────┘
     ▲          ▲
     │          │
┌────┴──────────┴────┐
│  MemoryService      │
│  ┌───────────────┐  │
│  │ShortTermMemory│  │  滑动窗口 · 内存+DB
│  │LongTermMemory │  │  向量检索 · 跨会话
│  └───────────────┘  │
└─────────────────────┘
         │
         ▼
┌─────────────────┐
│     MySQL        │  users / chat_sessions /
│   (持久化存储)    │  chat_messages / psychological_records
└─────────────────┘
```

### 2.2 请求处理流程

一次用户聊天的完整流程：

```
用户输入 → 前端 JS → POST /api/chat/send (JWT)
    → JwtAuthFilter 解析 token 获取 userId
    → ChatController.chat() 调用 ChatService
    → ChatService.chat() 调用 MindGuardAgent.process()
        → 构建 System Prompt（角色约束 + 工具描述）
        → 注入长期记忆（向量检索历史关键片段）
        → 注入短期记忆（当前会话滑动窗口上下文）
        → ReAct 循环（最多 5 轮）：
            → LLM 推理（Ollama qwen2.5:7b）
            → 解析是否需要调用工具
            → 是：执行工具 → 结果回传 LLM → 继续推理
            → 否：返回最终回答
        → 后处理：风险检测 → 自动写入 Excel / 发邮件
    → 返回 { "reply": "..." } → 前端渲染
```

---

## 三、各模块详解

### 3.1 Agent 核心 — MindGuardAgent

**文件**：`agent/MindGuardAgent.java`（350 行）

这是整个系统的大脑，采用 **ReAct（Reasoning + Acting）模式** 实现 Agentic RAG。

#### 3.1.1 ReAct 工作原理

ReAct 是一种让 LLM 交替进行「推理」和「行动」的 Agent 模式：

```
Thought: 用户说"最近很焦虑失眠"，我需要检索知识库给出专业建议
Action:  ragSearch(query="焦虑失眠缓解方法")
Observation: [知识库返回了3条相关建议...]
Thought: 我已获得知识库信息，可以综合回答了
Answer:  根据专业建议，你可以尝试以下方法...
```

#### 3.1.2 工具注册表

Agent 维护一个 `Map<String, ToolHandler>` 工具注册表，5 个工具通过函数式接口注册：

| 工具名 | 功能 | 调用条件 |
|--------|------|----------|
| `ragSearch` | 检索心理健康知识库 | 用户提出专业心理问题时 |
| `queryHistory` | 查询用户历史对话 | 需要了解用户过去情况时 |
| `sendEmail` | 发送高风险告警邮件 | 检测到自杀/严重抑郁时**必须**调用 |
| `writeExcel` | 写入心理状态 Excel 记录 | 每次对话结束后自动调用 |
| `bookCounselor` | 预约线下辅导员 | 判断学生需要线下咨询时 |

#### 3.1.3 工具调用决策机制

**方案选择与理由**：

| 方案 | 说明 | 是否采用 |
|------|------|----------|
| Spring AI 原生 Function Calling | 通过 `@Bean` 注册 Function，框架自动解析 LLM 输出的 tool_call | 否 |
| 关键词匹配 + 自动风险检测 | 解析 LLM 输出中的 `[call:toolName]` 标记，同时后处理自动检测风险关键词 | 是 |

**为什么不用原生 Function Calling**：Spring AI 1.0.0-M5 是里程碑预发布版，Ollama 的 Function Calling 支持不完整，在 qwen2.5:7b 上经常出现 tool_call 格式解析失败。关键词匹配方案虽然"笨"，但在当前模型能力下**实际可用性更高**，且配合后处理的自动风险检测（关键词匹配"自杀""自残"等），形成了双保险。

#### 3.1.4 多模型策略

通过 `mindguard.model.strategy` 配置切换：

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `ollama` | 默认使用 Ollama 本地模型 | **当前默认**，隐私优先，数据不出本机 |
| `openai` | 使用 OpenAI GPT-4o | 需要更强推理能力时 |
| `auto` | 优先 Ollama，失败自动回退 OpenAI | 平衡隐私与可靠性 |

**为什么默认 Ollama**：心理对话涉及敏感个人信息，使用本地模型确保数据不离开服务器，这是心理健康类应用的合规要求。

#### 3.1.5 自动风险检测（后处理）

Agent 在返回最终回答后，会执行 `autoDetectAndRecord()` 做二次保障：

```java
// 风险等级映射
自杀/自残/不想活/绝望 → CRITICAL → 自动发邮件 + 写 Excel
严重抑郁/恐慌发作/极度焦虑 → HIGH → 写 Excel
焦虑/失眠/担心 → MEDIUM → 写 Excel
其他 → LOW → 写 Excel
```

即使 LLM 没有在 ReAct 循环中主动调用 sendEmail 工具，后处理也能捕获高风险关键词并强制触发告警。

---

### 3.2 记忆系统 — 双层记忆架构

**文件**：`memory/MemoryService.java`、`ShortTermMemory.java`、`LongTermMemory.java`

这是本项目最核心的差异化设计——**短期记忆 + 长期记忆** 的双层架构，模拟人类记忆的工作方式。

#### 3.2.1 为什么需要双层记忆

| 方案 | 问题 |
|------|------|
| 只用上下文窗口 | 窗口有限（qwen2.5:7b 最多约 32K token），长对话会丢失早期信息 |
| 只用向量检索 | 每次都做语义检索，当前对话的连贯性无法保证 |
| 全存数据库 | 数据库能查但 LLM 读不了，需要向量化后才能语义匹配 |

双层记忆解决了这些矛盾：**短期记忆保证对话连贯，长期记忆实现跨会话关联**。

#### 3.2.2 短期记忆（ShortTermMemory）

**设计**：基于内存 + 数据库双重存储的滑动窗口

```
ConcurrentHashMap<sessionId, LinkedList<ChatMessage>>
     ↓ 超过 maxMessages (默认20条)
     ↓ 自动丢弃最早的消息
     ↓ 同时每条消息持久化到 chat_messages 表
```

**关键机制**：

- **滑动窗口**：内存中只保留最近 20 条消息，超出自动丢弃最早的，避免 token 溢出
- **双重存储**：内存用于快速读取，数据库用于持久化和恢复。应用重启后从数据库恢复最近 20 条
- **关键片段提取**：`extractKeyFragments()` 方法在会话结束时，筛选包含情绪/风险关键词的 assistant 消息，作为长期记忆的写入素材

**关键词列表**：`焦虑、抑郁、失眠、恐慌、自杀、绝望、建议、放松、咨询、风险、情绪、心理`

#### 3.2.3 长期记忆（LongTermMemory）

**设计**：基于 VectorStore 的语义检索

```
写入：关键片段 → TokenTextSplitter 切分 → 向量化 → 存入 VectorStore
查询：当前输入 → 向量相似度检索 → Top-K 匹配 → 按 userId 过滤
```

**关键机制**：

- **TokenTextSplitter 切分**：chunkSize=300, chunkOverlap=50，确保语义完整性
- **元数据标注**：每条记忆携带 `userId`、`sessionId`、`emotionLabel`、`riskLevel`、`timestamp`
- **用户隔离**：检索结果按 `userId` 元数据过滤，确保用户只能查到自己的历史

#### 3.2.4 记忆协调器（MemoryService）

负责短期和长期记忆的协同工作，核心方法：

**`buildEnhancedPrompt()`** — 构建 LLM 增强提示词：

```
System Prompt（角色约束 + 工具描述）
  +
长期记忆（语义检索的 Top-5 历史关键片段）
  +
短期记忆（当前会话滑动窗口上下文）
```

**`onSessionEnd()`** — 会话结束时的记忆归档：

```
1. 从短期记忆提取关键片段
2. 写入长期记忆（VectorStore）
3. 更新会话记录（风险等级、摘要）
4. 清除短期内存缓存
```

---

### 3.3 向量存储 — 方案演进与选择

**文件**：`config/VectorStoreConfig.java`

#### 3.3.1 方案演进过程

| 阶段 | 方案 | 结果 | 原因 |
|------|------|------|------|
| 初始设计 | ChromaDB + Spring AI Chroma Starter | 失败 | Spring AI 1.0.0-M5 使用 Chroma v1 API（`/api/v1/collections`），但 Chroma 1.5.x 已迁移到 v2 API，返回 410 Gone |
| 尝试降级 | 安装 Chroma 0.4.x（仍用 v1 API） | 失败 | Python 3.14 环境下 tokenizers 包编译失败 |
| 当前方案 | **SimpleVectorStore** | 成功 | Spring AI 内置的内存向量存储，零依赖启动 |

#### 3.3.2 当前实现

```java
@Bean
@ConditionalOnMissingBean(VectorStore.class)
public VectorStore vectorStore(EmbeddingModel embeddingModel) {
    return SimpleVectorStore.builder(embeddingModel).build();
}
```

**SimpleVectorStore 特点**：
- 纯内存存储，应用重启后数据丢失
- 适合开发测试阶段
- API 与 ChromaVectorStore 完全一致（都实现 `VectorStore` 接口）

**生产环境升级路径**：只需将 Chroma 版本兼容问题解决后，添加 `spring-ai-chroma-store-spring-boot-starter` 依赖，删除 `VectorStoreConfig`，Spring AI 会自动配置 ChromaVectorStore，**业务代码零改动**。

#### 3.3.3 Embedding 模型

当前使用 Ollama 的 qwen2.5:7b 同时作为 Chat 模型和 Embedding 模型。在 `application.yml` 中显式禁用了 OpenAI Embedding：

```yaml
spring:
  ai:
    openai:
      embedding:
        enabled: false   # 避免两个 EmbeddingModel Bean 冲突
```

**原因**：Spring AI 自动配置会同时注册 OpenAiEmbeddingModel 和 OllamaEmbeddingModel，导致 `EmbeddingModel` Bean 冲突。禁用 OpenAI Embedding 后，只用 Ollama 做 Embedding。

---

### 3.4 RAG 知识库 — DocumentService

**文件**：`service/DocumentService.java`

#### 3.4.1 知识库构建流程

```
知识库目录 (./knowledge/)
  ├── 心理健康指南.txt
  ├── 危机干预手册.md
  └── ...
       ↓
DocumentService.autoIngestKnowledgeBase()
       ↓ 扫描目录中的 .txt / .md / .pdf / .docx 文件
       ↓ 每个文件 → TokenTextSplitter(chunkSize=300, overlap=50)
       ↓ 切分后的 Document 携带元数据 (source, category, type="knowledge")
       ↓ vectorStore.add(chunks)
       ↓
  向量存储中的知识库
```

#### 3.4.2 RAG 检索

```java
public List<String> searchKnowledge(String query, int topK) {
    // 语义相似度检索
    List<Document> results = vectorStore.similaritySearch(
        SearchRequest.builder().query(query).topK(topK).build()
    );
    // 按 type="knowledge" 过滤，排除长期记忆中的对话片段
    return results.stream()
        .filter(doc -> "knowledge".equals(doc.getMetadata().get("type")))
        .limit(topK)
        .map(...)
        .toList();
}
```

**关键设计**：通过 `type` 元数据区分知识库文档和长期记忆对话片段，确保 RAG 检索只返回专业知识，不会把其他用户的对话历史误检索出来。

---

### 3.5 MCP 服务层 — McpClientService

**文件**：`service/McpClientService.java`

MCP（Model Context Protocol）是 Agent 与外部系统交互的协议标准。本项目中 McpClientService 封装了三个外部服务：

#### 3.5.1 邮件通知（sendEmail）

- 使用 Spring Boot 的 `JavaMailSender`
- 当检测到高风险学生时，自动发送告警邮件给配置的负责人
- 可通过 `mindguard.risk.high-risk-email-enabled` 开关控制
- 收件人通过 `mindguard.risk.high-risk-email-to` 配置（支持多个，逗号分隔）

#### 3.5.2 Excel 记录写入（writeExcel）

- 使用 Apache POI 操作 `.xlsx` 文件
- 文件不存在时自动创建（含表头：时间/学生ID/姓名/情绪标签/风险等级/对话摘要）
- 文件已存在时追加行
- **原子保存**：先写入 `.tmp` 临时文件，再 rename 替换原文件，避免写入中断导致文件损坏
- 路径通过 `mindguard.excel.path` 配置

#### 3.5.3 辅导员预约（bookCounselor）

- 当前为模拟实现，生成预约编号 `BK + 时间戳`
- 生产环境可对接真实的预约系统 API

**为什么不直接用 MCP 协议**：MCP 协议需要独立的 MCP Server 进程，当前阶段以本地实现为主，通过 `McpClientService` 统一封装，未来可以无缝替换为真正的 MCP Client → MCP Server 调用链路。

---

### 3.6 认证与安全 — Security + JWT

**文件**：`security/SecurityConfig.java`、`JwtAuthFilter.java`、`JwtUtil.java`、`service/AuthService.java`

#### 3.6.1 安全架构

```
请求 → JwtAuthFilter → 解析 Authorization: Bearer <token>
                          ↓
                    JwtUtil.validateToken()
                          ↓ 验证签名 + 过期时间
                    提取 userId / username / role
                          ↓
                    构造 UsernamePasswordAuthenticationToken
                    auth.setDetails(userId)  ← 关键：userId 存入 details
                          ↓
                    SecurityContextHolder 设置认证信息
                          ↓
                    Controller 通过 getCurrentUserId() 获取当前用户 ID
```

#### 3.6.2 JWT Token 结构

```
Header:  { "alg": "HS256" }
Payload: {
    "sub": "username",        // 用户名
    "userId": 1,              // 用户 ID
    "role": "USER",           // 角色
    "iat": 1716273600,        // 签发时间
    "exp": 1716360000         // 过期时间（默认24小时）
}
```

密钥通过 `jwt.secret` 配置，使用 HMAC-SHA256 签名。

#### 3.6.3 权限控制

| API 路径 | 权限要求 |
|----------|----------|
| `/`, `/index.html`, `/static/**` | 公开 |
| `/api/auth/**` | 公开（注册/登录） |
| `POST /api/chat/**` | 已认证用户 |
| `GET /api/chat/sessions` | 已认证用户（自动按 userId 过滤） |
| `GET /api/chat/records` | 已认证用户（自动按 userId 过滤） |
| `/api/admin/**` | ADMIN 角色 |
| `/api/knowledge/**` | ADMIN 角色 |

#### 3.6.4 用户数据隔离

**后端**：所有查询接口通过 `getCurrentUserId()` 获取当前登录用户的 ID，Repository 层使用 `findByUserIdOrderByXxx()` 方法，确保数据按用户隔离。`/api/chat/sessions/{id}/messages` 还额外校验会话是否属于当前用户，防止越权访问。

**前端**：聊天记录的 localStorage key 为 `mg_chats_{username}`，不同用户的数据完全隔离。心理记录从后端 API `/api/chat/records` 加载，天然按用户隔离。

---

### 3.7 数据模型

#### 3.7.1 实体关系图

```
┌──────────┐     1:N     ┌──────────────┐     1:N     ┌──────────────┐
│   User   │────────────→│ ChatSession  │────────────→│ ChatMessage  │
│──────────│             │──────────────│             │──────────────│
│ id       │             │ id           │             │ id           │
│ username │             │ userId       │             │ sessionId    │
│ password │             │ startTime    │             │ userId       │
│ role     │             │ endTime      │             │ role         │
│ realName │             │ finalRiskLevel│             │ content      │
│ email    │             │ summary      │             │ toolName     │
│ studentId│             │ archived     │             │ toolArgs     │
└──────────┘             └──────────────┘             │ createdAt    │
      │                                               └──────────────┘
      │ 1:N
      ▼
┌─────────────────────┐
│ PsychologicalRecord │
│─────────────────────│
│ id                  │
│ userId              │
│ sessionId           │
│ emotionLabel        │
│ riskLevel           │
│ conversationSummary │
│ keyPhrases          │
│ emailSent           │
│ recordedAt          │
└─────────────────────┘
```

#### 3.7.2 枚举类型

**Role（用户角色）**：
- `ADMIN` — 管理员，可上传知识库文档、查看所有记录
- `USER` — 普通学生，只能查看自己的数据

**RiskLevel（风险等级）**：
- `LOW` — 低风险
- `MEDIUM` — 中风险（焦虑、失眠等）
- `HIGH` — 高风险（严重抑郁、恐慌发作等）
- `CRITICAL` — 极高风险（自杀、自残倾向）

**EmotionLabel（情绪标签）**：
- `NORMAL` / `ANXIOUS` / `DEPRESSED` / `ANGRY` / `FEARFUL` / `HOPELESS` / `SUICIDAL`

---

### 3.8 前端界面

**文件**：`resources/static/index.html`（约 520 行，含 CSS + JS）

#### 3.8.1 界面结构

```
┌──────────────────────────────────────────┐
│  ┌──────────┐  ┌──────────────────────┐  │
│  │ Sidebar  │  │       Main           │  │
│  │──────────│  │──────────────────────│  │
│  │ Logo     │  │ Topbar (标题/风险徽章)│  │
│  │ +新对话  │  │──────────────────────│  │
│  │──────────│  │                      │  │
│  │ 对话历史 │  │   ChatArea (消息区)  │  │
│  │  · 对话1 │  │    欢迎页 / 消息列表 │  │
│  │  · 对话2 │  │    打字动画指示器    │  │
│  │──────────│  │──────────────────────│  │
│  │ 💬 对话  │  │ InputArea (输入框)   │  │
│  │ 📋 记录  │  │  [输入消息...] [发送]│  │
│  │ 👤 用户  │  └──────────────────────┘  │
│  └──────────┘                            │
└──────────────────────────────────────────┘
```

#### 3.8.2 核心功能

- **认证弹窗**：登录/注册切换，JWT token 存 localStorage
- **对话管理**：新建/切换/删除对话，数据按用户隔离
- **消息渲染**：用户右侧蓝色气泡、AI 左侧暗色气泡，打字动画
- **风险徽章**：顶部栏显示当前对话的风险等级（低/中/高/极高）
- **记录面板**：从后端 API 加载当前用户的心理监护记录
- **响应式**：768px 以下自动折叠侧边栏，出现菜单按钮

#### 3.8.3 布局修复说明

早期版本存在「对话多了输入框被挤出屏幕」的问题，原因：

- `#chatView` 没有设置 `flex` 布局和 `min-height: 0`
- `.main` 缺少 `overflow: hidden`

修复方案：

```css
.main { flex: 1; display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
#chatView { flex: 1; display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
.chat-area { flex: 1; overflow-y: auto; min-height: 0; }
.input-area { flex-shrink: 0; }
```

关键原理：flex 子元素需要 `min-height: 0` 才能正确收缩，`flex-shrink: 0` 保证输入框不被压缩。

---

## 四、API 接口清单

### 4.1 认证接口

| 方法 | 路径 | 权限 | 说明 |
|------|------|------|------|
| POST | `/api/auth/register` | 公开 | 注册，返回 JWT token |
| POST | `/api/auth/login` | 公开 | 登录，返回 JWT token |

**请求体**：
```json
{ "username": "student1", "password": "123456", "role": "USER", "realName": "张三" }
```

**响应**：
```json
{ "token": "eyJhbG...", "userId": 1, "username": "student1", "role": "USER" }
```

### 4.2 聊天接口

| 方法 | 路径 | 权限 | 说明 |
|------|------|------|------|
| POST | `/api/chat/send` | 已认证 | 发送消息，同步返回 AI 回复 |
| POST | `/api/chat/stream` | 已认证 | 发送消息，SSE 流式返回 |
| POST | `/api/chat/end-session` | 已认证 | 结束当前会话，触发记忆归档 |
| GET | `/api/chat/sessions` | 已认证 | 查询当前用户的会话列表 |
| GET | `/api/chat/sessions/{id}/messages` | 已认证 | 查询指定会话的消息（校验归属） |
| GET | `/api/chat/records` | 已认证 | 查询当前用户的心理监护记录 |

### 4.3 管理员接口

| 方法 | 路径 | 权限 | 说明 |
|------|------|------|------|
| POST | `/api/knowledge/upload` | ADMIN | 上传知识库文档 |
| POST | `/api/knowledge/reload` | ADMIN | 重载整个知识库 |
| GET | `/api/knowledge/search?query=xxx` | ADMIN | 测试知识库检索 |

---

## 五、技术方案选型与决策记录

### 5.1 Spring AI 版本选择

| 选项 | 版本 | 状态 | 选择 |
|------|------|------|------|
| GA 版 | 1.0.0 | 未发布（截至开发时） | 否 |
| 里程碑版 | 1.0.0-M5 | 已发布，功能基本完整 | 是 |
| 快照版 | 1.0.0-SNAPSHOT | 不稳定 | 否 |

**选择 M5 的理由**：GA 版尚未发布，M5 是最新可用版本，支持 Ollama 和 OpenAI 的 Chat/Embedding，功能满足需求。代价是部分 API 不稳定（如 Chroma 集成），需要 workaround。

### 5.2 Chroma vs SimpleVectorStore

| 维度 | ChromaDB | SimpleVectorStore |
|------|----------|-------------------|
| 数据持久化 | 独立进程，数据落盘 | 内存，重启丢失 |
| 语义检索 | 专业向量数据库，性能好 | 线性扫描，小规模可用 |
| 部署复杂度 | 需要 Docker / pip 安装 | 零依赖，开箱即用 |
| Spring AI 兼容 | M5 版与 Chroma v2 不兼容 | 完全兼容 |

**决策**：开发阶段用 SimpleVectorStore，生产环境升级 Chroma（等 Spring AI GA 版修复兼容性后）。

### 5.3 Ollama vs OpenAI

| 维度 | Ollama (qwen2.5:7b) | OpenAI (gpt-4o) |
|------|---------------------|-----------------|
| 数据隐私 | 数据不出本机 | 数据发送到 OpenAI 服务器 |
| 成本 | 免费 | 按 token 计费 |
| 推理质量 | 中等（7B 模型） | 优秀 |
| 部署要求 | 需要本地 GPU/至少 8G 内存 | 只需 API Key |
| Function Calling | 支持不完整 | 成熟可靠 |

**决策**：默认 Ollama（隐私优先），提供 `auto` 策略在 Ollama 不可用时回退 OpenAI。

### 5.4 关键词匹配 vs 原生 Function Calling

| 维度 | 关键词匹配 | Spring AI Function Calling |
|------|-----------|--------------------------|
| 实现复杂度 | 低 | 中 |
| 可靠性 | 高（确定性匹配） | 依赖模型输出格式 |
| 灵活性 | 低 | 高 |
| 当前模型兼容性 | 完全兼容 | Ollama 下不稳定 |

**决策**：当前用关键词匹配 + 后处理双保险。未来模型能力提升后切换到原生 Function Calling。

### 5.5 前端技术选型

| 选项 | 优势 | 劣势 | 选择 |
|------|------|------|------|
| React/Vue SPA | 组件化、生态丰富 | 需要构建工具、增加项目复杂度 | 否 |
| Thymeleaf 模板 | Spring Boot 原生支持 | 交互体验差，不适合实时聊天 | 否 |
| **单页 HTML + 原生 JS** | 零依赖、即开即用、部署简单 | 代码组织不如框架 | 是 |

**决策**：单页 HTML 方案。项目重点是后端 Agent 架构，前端够用即可，避免引入 Node.js 构建链路。

---

## 六、配置说明

`application.yml` 完整配置项：

```yaml
server:
  port: 8088                          # 服务端口

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mindguard?useUnicode=true&characterEncoding=utf8&serverTimezone=Asia/Shanghai
    username: root
    password:                         # MySQL 密码

  jpa:
    hibernate:
      ddl-auto: update                # 自动建表（开发环境）

  ai:
    openai:
      api-key: ${OPENAI_API_KEY}      # OpenAI API Key（环境变量）
      base-url: ${OPENAI_BASE_URL}    # 可替换为代理地址
      chat.options:
        model: gpt-4o
        temperature: 0.7
      embedding:
        enabled: false                # 禁用，避免 Bean 冲突

    ollama:
      base-url: http://localhost:11434
      chat.options:
        model: qwen2.5:7b
        temperature: 0.6
      embedding.options:
        model: qwen2.5:7b

  mail:                               # 邮件服务配置
    host: smtp.example.com
    port: 587
    username: ${MAIL_USERNAME}
    password: ${MAIL_PASSWORD}

jwt:
  secret: ${JWT_SECRET}               # JWT 签名密钥
  expiration: 86400000                 # 24小时（毫秒）

mindguard:
  model:
    strategy: ollama                  # ollama / openai / auto
  memory:
    short-term:
      max-messages: 20                # 短期记忆滑动窗口大小
    long-term:
      top-k: 5                        # 长期记忆检索 Top-K
      chunk-size: 300                 # 文本切分 chunk 大小
      chunk-overlap: 50               # 切分重叠字数
  risk:
    high-risk-email-enabled: true     # 是否启用高风险邮件
    high-risk-email-to: xxx@edu.cn    # 告警邮件收件人
  excel:
    path: ./data/psychological_records.xlsx  # Excel 记录路径
  knowledge:
    base-dir: ./knowledge/            # 知识库文档目录
```

---

## 七、部署与运行

### 7.1 环境依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| JDK | 17 | 运行 Spring Boot（Lombok 不兼容 JDK 26） |
| MySQL | 8.x | 数据持久化 |
| Ollama | 最新 | 本地大模型推理 |
| Maven | 3.9+ | 构建工具 |

### 7.2 启动步骤

```bash
# 1. 启动 MySQL，创建数据库
mysql -u root -p -e "CREATE DATABASE mindguard DEFAULT CHARACTER SET utf8mb4;"

# 2. 启动 Ollama，拉取模型
ollama serve
ollama pull qwen2.5:7b

# 3. 启动应用
export JAVA_HOME=/path/to/jdk-17
mvn spring-boot:run

# 4. 访问
open http://localhost:8088
```

### 7.3 生产环境打包

```bash
mvn clean package -DskipTests
java -jar target/mindguard-1.0.0.jar --spring.config.additional-location=/path/to/application-prod.yml
```

---

## 八、已知限制与未来规划

| 限制 | 原因 | 规划 |
|------|------|------|
| 向量存储重启丢失 | 使用 SimpleVectorStore（内存） | 升级 ChromaDB 或 Milvus |
| Function Calling 不可靠 | Ollama qwen2.5:7b 工具调用格式不稳定 | 升级模型或切换到原生 Function Calling |
| 流式输出为模拟实现 | Spring AI M5 对 Ollama 流式支持有限 | 升级 Spring AI GA 版 |
| 辅导员预约为模拟 | 未对接真实预约系统 | 对接学校教务系统 API |
| 无管理后台前端 | 当前只有聊天界面 | 新增 Admin Dashboard |
| 长期记忆检索效率 | SimpleVectorStore 线性扫描 | 切换专业向量数据库 |
