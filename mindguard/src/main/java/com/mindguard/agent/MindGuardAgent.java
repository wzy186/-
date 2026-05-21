package com.mindguard.agent;

import com.mindguard.memory.MemoryService;
import com.mindguard.tools.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * MindGuard Agent 核心：ReAct 模式的 Agentic RAG 系统。
 *
 * 工作流程：
 * 1. 接收用户输入
 * 2. 构建 Prompt（System + 长期记忆 + 短期记忆 + 工具描述）
 * 3. LLM 自主推理（Thought → Action → Observation 循环）
 * 4. 通过 Function Calling 动态调用工具
 * 5. 工具结果回传 LLM，继续推理或输出最终回答
 *
 * 5 个可用工具：
 * - RagSearchTool: 检索知识库
 * - QueryHistoryTool: 查询历史对话
 * - SendEmailTool: 发送高风险告警邮件
 * - WriteExcelTool: 写入心理状态记录
 * - BookCounselorTool: 预约辅导员
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MindGuardAgent {

    private final OpenAiChatModel openAiChatModel;
    private final OllamaChatModel ollamaChatModel;
    private final MemoryService memoryService;
    private final RagSearchTool ragSearchTool;
    private final QueryHistoryTool queryHistoryTool;
    private final SendEmailTool sendEmailTool;
    private final WriteExcelTool writeExcelTool;
    private final BookCounselorTool bookCounselorTool;

    @Value("${mindguard.model.strategy:ollama}")
    private String modelStrategy;

    // 工具注册表
    private final Map<String, ToolHandler> toolRegistry = new LinkedHashMap<>();

    /**
     * 工具处理器接口
     */
    @FunctionalInterface
    interface ToolHandler {
        String handle(Long userId, String username, Map<String, Object> args);
    }

    /**
     * 初始化工具注册表
     */
    public void init() {
        toolRegistry.put("ragSearch", (userId, username, args) ->
                ragSearchTool.search((String) args.getOrDefault("query", "")));

        toolRegistry.put("queryHistory", (userId, username, args) ->
                queryHistoryTool.query(userId, (String) args.getOrDefault("query", "")));

        toolRegistry.put("sendEmail", (userId, username, args) ->
                sendEmailTool.sendAlert(userId, username,
                        (String) args.getOrDefault("riskLevel", "HIGH"),
                        (String) args.getOrDefault("summary", "")));

        toolRegistry.put("writeExcel", (userId, username, args) ->
                writeExcelTool.writeRecord(userId, username,
                        (String) args.getOrDefault("emotionLabel", "NORMAL"),
                        (String) args.getOrDefault("riskLevel", "LOW"),
                        (String) args.getOrDefault("summary", "")));

        toolRegistry.put("bookCounselor", (userId, username, args) ->
                bookCounselorTool.book(userId, username,
                        (String) args.getOrDefault("reason", "")));
    }

    /**
     * ReAct 主循环：处理用户消息
     *
     * @param userId    用户 ID
     * @param username  用户名
     * @param sessionId 会话 ID
     * @param userRole  用户角色（ADMIN / USER）
     * @param userInput 用户输入
     * @return 最终回复
     */
    public String process(Long userId, String username, Long sessionId,
                          String userRole, String userInput) {
        if (toolRegistry.isEmpty()) {
            init();
        }

        // 1. 构建 System Prompt（含角色约束）
        String systemPrompt = buildSystemPrompt(userRole, username);

        // 2. 构建增强 Prompt（注入长期记忆 + 短期记忆）
        String enhancedPrompt = memoryService.buildEnhancedPrompt(
                userId, sessionId, systemPrompt, userInput);

        // 3. 构建消息列表
        List<Message> messages = new ArrayList<>();
        messages.add(new SystemMessage(enhancedPrompt));

        // 加载短期记忆作为对话历史
        String shortTermContext = memoryService.getCurrentContext(sessionId);
        if (!shortTermContext.isEmpty()) {
            // 解析并添加历史消息
            for (String line : shortTermContext.split("\n")) {
                if (line.startsWith("用户: ")) {
                    messages.add(new UserMessage(line.substring(4)));
                } else if (line.startsWith("助手: ")) {
                    messages.add(new AssistantMessage(line.substring(4)));
                }
            }
        }

        // 当前用户输入
        messages.add(new UserMessage(userInput));

        // 4. ReAct 循环
        int maxIterations = 5;
        for (int i = 0; i < maxIterations; i++) {
            log.info("[ReAct] 迭代 {}/{}, userId={}, tools_available={}",
                    i + 1, maxIterations, userId, toolRegistry.keySet());

            // 调用 LLM
            ChatResponse response = callLlm(messages);
            String reply = response.getResult().getOutput().getContent();

            // 检查 LLM 是否想调用工具
            ToolCallDecision decision = parseToolCall(reply);

            if (decision == null || !decision.wantsToCall) {
                // LLM 不想调用工具，直接返回回答
                // 记录到短期记忆
                memoryService.recordMessage(sessionId, userId, "user", userInput, null, null);
                memoryService.recordMessage(sessionId, userId, "assistant", reply, null, null);

                // 后处理：自动检测风险并触发写入
                autoDetectAndRecord(userId, username, sessionId, reply);

                return reply;
            }

            // 5. 执行工具调用
            log.info("[ReAct] LLM 决定调用工具: {}", decision.toolName);
            String toolResult = executeTool(userId, username, decision);

            // 6. 工具结果回传 LLM
            messages.add(new AssistantMessage(reply));
            messages.add(new UserMessage("[工具 " + decision.toolName + " 的执行结果]\n" + toolResult));

            // 记录到短期记忆
            memoryService.recordMessage(sessionId, userId, "tool_call", null,
                    decision.toolName, decision.args.toString());
            memoryService.recordMessage(sessionId, userId, "tool_response", toolResult, null, null);
        }

        // 超过最大迭代次数，返回最后结果
        String fallback = "抱歉，我需要更多时间来处理您的问题。请稍后再试或联系辅导员。";
        memoryService.recordMessage(sessionId, userId, "user", userInput, null, null);
        memoryService.recordMessage(sessionId, userId, "assistant", fallback, null, null);
        return fallback;
    }

    /**
     * 构建 System Prompt（含角色权限约束）
     */
    private String buildSystemPrompt(String userRole, String username) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是 MindGuard AI 心理监护助手，专门为高校学生提供心理健康支持和咨询服务。\n\n");

        sb.append("## 你的核心能力\n");
        sb.append("1. 倾听和理解学生的心理困扰\n");
        sb.append("2. 提供基于知识库的专业心理健康建议\n");
        sb.append("3. 识别高风险信号并自动通知相关负责人\n");
        sb.append("4. 帮助学生预约线下心理咨询\n\n");

        sb.append("## 可用工具\n");
        sb.append("- ragSearch: 检索心理健康知识库（当用户提出专业心理问题时使用）\n");
        sb.append("- queryHistory: 查询用户的历史对话（当需要了解用户过去情况时使用）\n");
        sb.append("- sendEmail: 发送高风险告警邮件（当检测到自杀倾向、严重抑郁等高风险时必须使用）\n");
        sb.append("- writeExcel: 将心理状态写入Excel记录（每次对话结束时应使用）\n");
        sb.append("- bookCounselor: 预约辅导员（当判断学生需要线下咨询时使用）\n\n");

        // 角色权限约束
        sb.append("## 当前角色权限\n");
        if ("ADMIN".equals(userRole)) {
            sb.append("你是管理员，可以：\n");
            sb.append("- 访问所有知识库内容\n");
            sb.append("- 查看所有学生的心理记录\n");
            sb.append("- 使用全部工具\n");
        } else {
            sb.append("你是普通用户（学生），只能：\n");
            sb.append("- 访问公开的心理健康知识\n");
            sb.append("- 查询自己的历史对话\n");
            sb.append("- 不能查看其他学生的心理记录\n");
        }

        sb.append("\n## 安全规则\n");
        sb.append("- 当检测到自杀倾向、自残意图时，必须立即调用 sendEmail 工具\n");
        sb.append("- 对话结束后应调用 writeExcel 记录心理状态\n");
        sb.append("- 不要编造医学诊断，始终基于知识库内容回答\n");
        sb.append("- 对学生保持温暖、共情、非评判的态度\n");

        return sb.toString();
    }

    /**
     * 调用 LLM（根据策略选择模型）
     */
    private ChatResponse callLlm(List<Message> messages) {
        Prompt prompt = new Prompt(messages);
        return switch (modelStrategy) {
            case "openai" -> openAiChatModel.call(prompt);
            case "ollama" -> ollamaChatModel.call(prompt);
            case "auto" -> {
                // 自动策略：敏感数据用 Ollama 本地模型，复杂推理用 OpenAI
                try {
                    yield ollamaChatModel.call(prompt);
                } catch (Exception e) {
                    log.warn("Ollama 调用失败，回退到 OpenAI", e);
                    yield openAiChatModel.call(prompt);
                }
            }
            default -> ollamaChatModel.call(prompt);
        };
    }

    /**
     * 解析 LLM 输出中的工具调用意图
     * 实际生产中使用 SpringAI Function Calling 自动解析，
     * 此处为简化版本，通过关键词匹配模拟
     */
    private ToolCallDecision parseToolCall(String llmOutput) {
        if (llmOutput == null) return null;

        String lower = llmOutput.toLowerCase();

        // 检测工具调用意图
        if (lower.contains("[call:") || lower.contains("调用工具:")) {
            ToolCallDecision decision = new ToolCallDecision();
            decision.wantsToCall = true;

            // 解析工具名和参数
            for (String toolName : toolRegistry.keySet()) {
                if (lower.contains(toolName.toLowerCase()) || lower.contains(toolName)) {
                    decision.toolName = toolName;
                    decision.args = new HashMap<>();
                    // 简化参数解析
                    if (llmOutput.contains("query=")) {
                        decision.args.put("query", extractParam(llmOutput, "query"));
                    }
                    if (llmOutput.contains("riskLevel=")) {
                        decision.args.put("riskLevel", extractParam(llmOutput, "riskLevel"));
                    }
                    if (llmOutput.contains("reason=")) {
                        decision.args.put("reason", extractParam(llmOutput, "reason"));
                    }
                    return decision;
                }
            }
        }

        // 自动检测高风险关键词 → 触发工具调用
        if (lower.contains("自杀") || lower.contains("自残") || lower.contains("不想活")) {
            ToolCallDecision decision = new ToolCallDecision();
            decision.wantsToCall = true;
            decision.toolName = "sendEmail";
            decision.args = Map.of("riskLevel", "CRITICAL", "summary", llmOutput);
            return decision;
        }

        return null;
    }

    private String extractParam(String text, String key) {
        String marker = key + "=";
        int start = text.indexOf(marker);
        if (start == -1) return "";
        start += marker.length();
        int end = text.indexOf(",", start);
        if (end == -1) end = text.indexOf("]", start);
        if (end == -1) end = text.length();
        return text.substring(start, end).trim();
    }

    /**
     * 执行工具调用
     */
    private String executeTool(Long userId, String username, ToolCallDecision decision) {
        ToolHandler handler = toolRegistry.get(decision.toolName);
        if (handler == null) {
            return "工具不存在: " + decision.toolName;
        }
        try {
            return handler.handle(userId, username, decision.args);
        } catch (Exception e) {
            log.error("工具执行失败: {}", decision.toolName, e);
            return "工具执行失败: " + e.getMessage();
        }
    }

    /**
     * 自动风险检测和记录写入（后处理）
     */
    private void autoDetectAndRecord(Long userId, String username, Long sessionId, String reply) {
        String lower = reply.toLowerCase();
        String riskLevel = "LOW";
        String emotionLabel = "NORMAL";

        // 风险等级检测
        if (lower.contains("自杀") || lower.contains("自残") || lower.contains("不想活") || lower.contains("绝望")) {
            riskLevel = "CRITICAL";
            emotionLabel = "SUICIDAL";
            sendEmailTool.sendAlert(userId, username, riskLevel, reply);
        } else if (lower.contains("严重抑郁") || lower.contains("恐慌发作") || lower.contains("极度焦虑")) {
            riskLevel = "HIGH";
            emotionLabel = "DEPRESSED";
        } else if (lower.contains("焦虑") || lower.contains("失眠") || lower.contains("担心")) {
            riskLevel = "MEDIUM";
            emotionLabel = "ANXIOUS";
        }

        // 自动写入 Excel
        writeExcelTool.writeRecord(userId, username, emotionLabel, riskLevel, reply);
    }

    /**
     * 工具调用决策对象
     */
    private static class ToolCallDecision {
        boolean wantsToCall = false;
        String toolName;
        Map<String, Object> args;
    }
}
