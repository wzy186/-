package com.dss.ai.agent;

import com.dss.ai.client.LlmClient;
import com.dss.ai.dto.ChatMessage;
import com.dss.ai.dto.ToolDef;
import com.dss.ai.tool.ToolRegistry;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * Agent:ReAct 循环驱动的多步规划。
 * <p>
 * 流程:
 * 1. 用户问题 + 系统提示 + 工具列表 → LLM
 * 2. LLM 返回 tool_calls → 执行工具 → 结果塞回消息
 * 3. 再次调 LLM,它可能继续调工具或给最终答案
 * 4. 最多循环 MAX_STEPS 次,防死循环
 * <p>
 * 每一步的思考/工具调用通过 onStep 回调暴露给前端(可视化)。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class Agent {

    private final LlmClient llm;
    private final ToolRegistry toolRegistry;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private static final int MAX_STEPS = 6;

    /**
     * @param userQuery 用户问题
     * @param onStep    每一步(思考/工具调用/结果)的回调,前端可视化用
     * @param onToken   最终答案的流式 token 回调
     */
    public void run(String userQuery, Long userId, Consumer2<String> onStep, java.util.function.Consumer<String> onToken) {
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(buildSystemPrompt(userId)));
        messages.add(ChatMessage.user(userQuery));

        List<ToolDef> tools = toolRegistry.allToolDefs();

        for (int step = 1; step <= MAX_STEPS; step++) {
            LlmClient.ChatResult result = llm.chat(messages, tools);

            // 如果 LLM 要调工具
            if (result.hasToolCall()) {
                // 先把 assistant 的 tool_calls 消息加入历史
                messages.add(buildAssistantWithToolCalls(result));

                // 逐个执行工具
                for (Map<String, Object> tc : result.toolCalls()) {
                    Map<String, Object> function = (Map<String, Object>) tc.get("function");
                    String toolName = (String) function.get("name");
                    String argsJson = (String) function.get("arguments");
                    String toolCallId = (String) tc.get("id");

                    Map<String, Object> args = parseArgs(argsJson);
                    onStep.accept("🔧 调用工具: " + toolName + " 参数: " + argsJson);
                    log.info("Agent step {} 调工具: {} {}", step, toolName, argsJson);

                    String toolResult = toolRegistry.execute(toolName, args);
                    onStep.accept("📋 结果: " + truncate(toolResult, 200));

                    messages.add(ChatMessage.tool(toolName, toolCallId, toolResult));
                }
                continue; // 继续循环让 LLM 看工具结果
            }

            // 没有工具调用 = 最终答案
            String answer = result.content() == null ? "(无回复)" : result.content();
            onStep.accept("💡 最终回答");
            // 流式输出最终答案(这里用分段模拟,真流式需 LLM stream + 工具循环,较复杂)
            streamOut(answer, onToken);
            return;
        }

        // 超过最大步数
        onStep.accept("⚠️ 达到最大步数 " + MAX_STEPS + ",停止");
        streamOut("抱歉,这个问题我尝试了多步但仍未完成,请换个问法。", onToken);
    }

    private String buildSystemPrompt(Long userId) {
        return """
            你是兜省省的 AI 省钱助手。用户当前登录 ID: %s。
            你可以调用工具查询商铺、优惠券、红包雨,甚至帮用户秒杀下单、抢红包。
            规则:
            1. 涉及实时数据(商铺/券/红包)时,必须调工具查,不要凭空编造。
            2. 帮用户下单/抢红包前,先告知用户你将操作,并确认(除非用户明确说"直接帮我抢")。
            3. 回答简洁友好,用中文。
            4. 推荐时给出具体商铺名和价格,不要泛泛而谈。
            """.formatted(userId == null ? "未登录" : userId);
    }

    private ChatMessage buildAssistantWithToolCalls(LlmClient.ChatResult result) {
        // OpenAI 格式:assistant 消息带 tool_calls 数组
        ChatMessage m = new ChatMessage();
        m.setRole("assistant");
        m.setContent(result.content());
        m.setToolCalls(result.toolCalls());
        return m;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseArgs(String json) {
        if (json == null || json.isBlank()) return Map.of();
        try { return objectMapper.readValue(json, Map.class); }
        catch (Exception e) { log.warn("参数解析失败: {}", json); return Map.of(); }
    }

    private String truncate(String s, int n) {
        if (s == null) return "";
        return s.length() > n ? s.substring(0, n) + "..." : s;
    }

    private void streamOut(String text, java.util.function.Consumer<String> onToken) {
        // 按 char 流式输出(模拟打字机)。真流式见 LlmClient.chatStream。
        for (int i = 0; i < text.length(); i++) {
            onToken.accept(String.valueOf(text.charAt(i)));
        }
    }

    /** 双参数 Consumer 兼容(简单用 Consumer<String>) */
    public interface Consumer2<T> { void accept(T t); }
}
