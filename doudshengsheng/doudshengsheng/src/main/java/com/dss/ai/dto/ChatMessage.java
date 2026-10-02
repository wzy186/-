package com.dss.ai.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * LLM 请求/响应 DTO(OpenAI 兼容格式)。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ChatMessage {
    private String role;    // system / user / assistant / tool
    private String content;
    private List<Map<String, Object>> toolCalls;  // assistant 发起的工具调用
    private String toolCallId; // tool 角色消息的关联 id
    private String name;       // tool 名

    public ChatMessage(String role, String content) {
        this.role = role;
        this.content = content;
    }

    /** 构造 user 消息 */
    public static ChatMessage user(String content) {
        return new ChatMessage("user", content);
    }

    /** 构造 system 消息 */
    public static ChatMessage system(String content) {
        return new ChatMessage("system", content);
    }

    /** 构造 assistant 消息 */
    public static ChatMessage assistant(String content) {
        return new ChatMessage("assistant", content);
    }

    /** 构造 tool 结果消息 */
    public static ChatMessage tool(String name, String toolCallId, String content) {
        ChatMessage m = new ChatMessage();
        m.role = "tool";
        m.name = name;
        m.toolCallId = toolCallId;
        m.content = content;
        return m;
    }
}
