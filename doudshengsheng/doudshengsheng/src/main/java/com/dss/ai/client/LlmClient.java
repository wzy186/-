package com.dss.ai.client;

import com.dss.ai.config.AiProperties;
import com.dss.ai.dto.ChatMessage;
import com.dss.ai.dto.ToolDef;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * LLM 客户端:调用 DeepSeek(OpenAI 兼容格式)。
 * 同步用 RestClient,流式用 OkHttp 手动读 SSE(逐 token 推给前端)。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class LlmClient {

    private final AiProperties props;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build();

    /**
     * 同步对话。
     *
     * @param messages 消息列表
     * @param tools    可用工具(可为 null)
     * @return assistant 回复(纯文本 或 tool_calls)
     */
    @SuppressWarnings("unchecked")
    public ChatResult chat(List<ChatMessage> messages, List<ToolDef> tools) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", props.getDeepseek().getModel());
        body.put("messages", messages.stream().map(this::toMap).toList());
        body.put("temperature", 0.3);
        if (tools != null && !tools.isEmpty()) {
            body.put("tools", tools);
            body.put("tool_choice", "auto");
        }

        try {
            String json = objectMapper.writeValueAsString(body);
            Request req = new Request.Builder()
                    .url(props.getDeepseek().getBaseUrl() + "/chat/completions")
                    .header("Authorization", "Bearer " + props.getDeepseek().getApiKey())
                    .post(RequestBody.create(json, MediaType.parse("application/json")))
                    .build();
            try (Response resp = httpClient.newCall(req).execute()) {
                String respBody = resp.body() == null ? "" : resp.body().string();
                if (!resp.isSuccessful()) {
                    throw new RuntimeException("LLM 返回 " + resp.code() + ": " + respBody);
                }
                Map<String, Object> respMap = objectMapper.readValue(respBody, Map.class);
                Map<String, Object> choice = (Map<String, Object>) ((List<?>) respMap.get("choices")).get(0);
                Map<String, Object> msg = (Map<String, Object>) choice.get("message");
                String content = msg.get("content") == null ? null : msg.get("content").toString();
                List<Map<String, Object>> toolCalls = (List<Map<String, Object>>) msg.get("tool_calls");
                String reasoning = msg.get("reasoning_content") == null ? null : msg.get("reasoning_content").toString();
                return new ChatResult(content, toolCalls, reasoning);
            }
        } catch (Exception e) {
            log.error("LLM 调用失败: {}", e.getMessage());
            throw new RuntimeException("LLM 调用失败: " + e.getMessage(), e);
        }
    }

    /**
     * 流式对话(SSE):逐 token 调用 onToken。
     * DeepSeek 流式格式:每个 chunk 的 choices[0].delta.content
     */
    public void chatStream(List<ChatMessage> messages, Consumer<String> onToken) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", props.getDeepseek().getModel());
        body.put("messages", messages.stream().map(this::toMap).toList());
        body.put("temperature", 0.3);
        body.put("stream", true);

        try {
            String json = objectMapper.writeValueAsString(body);
            Request req = new Request.Builder()
                    .url(props.getDeepseek().getBaseUrl() + "/chat/completions")
                    .header("Authorization", "Bearer " + props.getDeepseek().getApiKey())
                    .header("Accept", "text/event-stream")
                    .post(RequestBody.create(json, MediaType.parse("application/json")))
                    .build();
            try (Response resp = httpClient.newCall(req).execute()) {
                if (!resp.isSuccessful() || resp.body() == null) {
                    throw new RuntimeException("流式返回 " + resp.code());
                }
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(resp.body().byteStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (!line.startsWith("data: ")) continue;
                        String data = line.substring(6).trim();
                        if ("[DONE]".equals(data)) break;
                        try {
                            Map<String, Object> chunk = objectMapper.readValue(data, Map.class);
                            List<?> choices = (List<?>) chunk.get("choices");
                            if (choices == null || choices.isEmpty()) continue;
                            Map<String, Object> delta = (Map<String, Object>) ((Map<String, Object>) choices.get(0)).get("delta");
                            if (delta != null && delta.get("content") != null) {
                                onToken.accept(delta.get("content").toString());
                            }
                        } catch (Exception ignored) {
                            // 单 chunk 解析失败不影响整体
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("流式调用失败: {}", e.getMessage());
            throw new RuntimeException(e);
        }
    }

    private Map<String, Object> toMap(ChatMessage m) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("role", m.getRole());
        // 注意:assistant 带 tool_calls 时 content 可能为 null,但 OpenAI 要求有 content 字段
        if (m.getContent() != null) {
            map.put("content", m.getContent());
        } else if ("assistant".equals(m.getRole()) && m.getToolCalls() != null) {
            map.put("content", null);
        }
        if (m.getToolCalls() != null) map.put("tool_calls", m.getToolCalls());
        if (m.getToolCallId() != null) map.put("tool_call_id", m.getToolCallId());
        if (m.getName() != null) map.put("name", m.getName());
        return map;
    }

    /** LLM 返回结果 */
    public record ChatResult(String content, List<Map<String, Object>> toolCalls, String reasoning) {
        public boolean hasToolCall() {
            return toolCalls != null && !toolCalls.isEmpty();
        }
    }
}
