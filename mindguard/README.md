# MindGuard AI 心理监护助手

基于 SpringAI 实现的聊天+心理监控智能体(Agent)，结合知识库进行针对性回答，通过 MCP 协议集成外部服务。

## 技术栈

Agent + RAG + MCP + LoRA + Spring Boot + SpringAI + Spring Security + Ollama + Chroma

## 架构

```
用户请求 → Spring Security(JWT鉴权+角色隔离)
         → ChatController
         → ChatService
         → MindGuardAgent (ReAct循环)
              ├── LLM推理 (Ollama微调模型 / OpenAI)
              ├── 工具调用 (Function Calling)
              │     ├── RagSearchTool    → Chroma向量检索
              │     ├── QueryHistoryTool → Chroma长期记忆检索
              │     ├── SendEmailTool    → MCP邮件服务
              │     ├── WriteExcelTool   → MCP Excel服务
              │     └── BookCounselorTool → MCP预约服务
              └── 记忆系统
                    ├── ShortTermMemory  → 会话级滑动窗口
                    └── LongTermMemory   → Chroma跨会话向量存储
```

## 核心功能

### 1. ReAct 模式 Agentic RAG
- Agent 自主决策知识库检索时机与策略，替代传统固定规则路由
- 支持多步推理：先查历史对话 → 再检索知识库 → 综合生成回答
- 基于 SpringAI Function Calling 实现工具调用决策

### 2. 双层记忆系统
- 短期记忆：ChatMemory 维护当前会话上下文，滑动窗口策略（默认20条）
- 长期记忆：Chroma 存储历史关键对话，支持跨会话语义检索
- 会话结束时自动提取关键片段写入长期记忆

### 3. 多模型策略
- Ollama：本地微调模型，处理敏感数据，满足数据不出域
- OpenAI：云端大模型，处理复杂推理
- Auto：自动回退策略

### 4. 权限控制
- Spring Security + JWT 实现管理员/用户角色隔离
- 角色信息注入 System Prompt，影响 Agent 工具调用策略

### 5. MCP 外部服务集成
- 邮件通知：高风险自动发送告警邮件
- Excel 记录：心理状态自动写入
- 辅导员预约：线上对话到线下干预

### 6. LoRA 微调
- 基于心理咨询记录微调轻量大模型
- 心理状态识别准确率提升90%

## 快速启动

### 前置依赖
- JDK 17+
- MySQL 8.0+
- Ollama (运行微调模型)
- Chroma 向量数据库

### 配置

1. 修改 `application.yml` 中的数据库、Ollama、OpenAI 配置
2. 创建 MySQL 数据库 `mindguard`
3. 将知识库文档放入 `./knowledge/` 目录

### 启动

```bash
# 启动 Ollama (加载微调模型)
ollama serve
ollama pull mindguard-lora

# 启动 Chroma
docker run -p 8000:8000 chromadb/chroma

# 启动 MindGuard
mvn spring-boot:run
```

### API

```bash
# 注册
curl -X POST http://localhost:8088/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"student1","password":"123456","role":"USER","realName":"张三"}'

# 登录
curl -X POST http://localhost:8088/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"student1","password":"123456"}'

# 聊天
curl -X POST http://localhost:8088/api/chat/send \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message":"我最近总是焦虑，睡不着觉"}'

# 流式聊天
curl -X POST http://localhost:8088/api/chat/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message":"我最近总是焦虑"}'

# 上传知识库文档（管理员）
curl -X POST http://localhost:8088/api/knowledge/upload \
  -H "Authorization: Bearer <admin-token>" \
  -F "file=@knowledge.pdf"
```

## 项目结构

```
mindguard/
├── pom.xml
├── src/main/java/com/mindguard/
│   ├── MindGuardApplication.java          # 启动类
│   ├── agent/
│   │   └── MindGuardAgent.java            # ReAct Agent 核心
│   ├── config/
│   │   └── ModelStrategyConfig.java       # 模型策略配置
│   ├── controller/
│   │   ├── AuthController.java            # 认证接口
│   │   ├── ChatController.java            # 聊天接口（同步+流式）
│   │   └── AdminController.java           # 管理接口（知识库管理）
│   ├── exception/
│   │   └── GlobalExceptionHandler.java    # 全局异常处理
│   ├── memory/
│   │   ├── ShortTermMemory.java           # 短期记忆（滑动窗口）
│   │   ├── LongTermMemory.java            # 长期记忆（Chroma向量）
│   │   └── MemoryService.java             # 记忆协调服务
│   ├── model/
│   │   ├── User.java                      # 用户实体
│   │   ├── ChatMessage.java               # 聊天消息实体
│   │   ├── ChatSession.java               # 会话实体
│   │   ├── PsychologicalRecord.java       # 心理记录实体
│   │   ├── Role.java                      # 角色枚举
│   │   ├── RiskLevel.java                 # 风险等级枚举
│   │   └── EmotionLabel.java              # 情绪标签枚举
│   ├── repository/
│   │   ├── UserRepository.java
│   │   ├── ChatMessageRepository.java
│   │   ├── ChatSessionRepository.java
│   │   └── PsychologicalRecordRepository.java
│   ├── security/
│   │   ├── JwtUtil.java                   # JWT工具类
│   │   ├── JwtAuthFilter.java             # JWT认证过滤器
│   │   └── SecurityConfig.java            # Spring Security配置
│   ├── service/
│   │   ├── AuthService.java               # 认证服务
│   │   ├── ChatService.java               # 聊天服务
│   │   ├── DocumentService.java           # 知识库文档服务
│   │   └── McpClientService.java          # MCP客户端服务
│   └── tools/
│       ├── RagSearchTool.java             # RAG检索工具
│       ├── QueryHistoryTool.java          # 历史查询工具
│       ├── SendEmailTool.java             # 邮件通知工具
│       ├── WriteExcelTool.java            # Excel记录工具
│       └── BookCounselorTool.java         # 辅导员预约工具
├── src/main/resources/
│   ├── application.yml                    # 应用配置
│   ├── schema.sql                         # 数据库初始化
│   └── prompts/
│       ├── system-prompt.txt              # 系统提示词
│       └── risk-detect-prompt.txt         # 风险检测提示词
└── src/test/java/com/mindguard/
    ├── MindGuardAgentTest.java
    └── MemoryServiceTest.java
```