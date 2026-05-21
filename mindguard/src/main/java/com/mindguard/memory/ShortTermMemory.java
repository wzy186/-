package com.mindguard.memory;

import com.mindguard.model.ChatMessage;
import com.mindguard.repository.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 短期记忆：基于内存的会话级消息存储，滑动窗口策略控制上下文长度。
 * - 维护当前会话的完整消息流（user / assistant / tool_call / tool_response）
 * - 超出窗口的早期消息自动丢弃，但关键信息在会话结束时被提取到长期记忆
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ShortTermMemory {

    private final ChatMessageRepository chatMessageRepository;

    @Value("${mindguard.memory.short-term.max-messages:20}")
    private int maxMessages;

    // sessionId -> 消息列表
    private final ConcurrentHashMap<Long, LinkedList<ChatMessage>> sessionMessages = new ConcurrentHashMap<>();

    /**
     * 追加一条消息到短期记忆
     */
    public void addMessage(Long sessionId, Long userId, String role, String content,
                           String toolName, String toolArgs) {
        ChatMessage msg = ChatMessage.builder()
                .sessionId(sessionId)
                .userId(userId)
                .role(role)
                .content(content)
                .toolName(toolName)
                .toolArgs(toolArgs)
                .createdAt(LocalDateTime.now())
                .build();

        // 持久化到数据库
        chatMessageRepository.save(msg);

        // 内存缓存
        sessionMessages.computeIfAbsent(sessionId, k -> new LinkedList<>()).addLast(msg);

        // 滑动窗口：超出限制时移除最早的消息
        LinkedList<ChatMessage> messages = sessionMessages.get(sessionId);
        while (messages.size() > maxMessages) {
            messages.removeFirst();
        }
    }

    /**
     * 获取会话的所有短期记忆（格式化为 LLM 可理解的上下文）
     */
    public List<ChatMessage> getMessages(Long sessionId) {
        // 优先从内存取，否则从数据库恢复
        LinkedList<ChatMessage> cached = sessionMessages.get(sessionId);
        if (cached != null && !cached.isEmpty()) {
            return new ArrayList<>(cached);
        }
        List<ChatMessage> fromDb = chatMessageRepository
                .findTop20BySessionIdOrderByCreatedAtDesc(sessionId);
        Collections.reverse(fromDb);
        sessionMessages.put(sessionId, new LinkedList<>(fromDb));
        return fromDb;
    }

    /**
     * 将短期记忆格式化为 Prompt 上下文字符串
     */
    public String formatAsContext(Long sessionId) {
        List<ChatMessage> messages = getMessages(sessionId);
        StringBuilder sb = new StringBuilder();
        for (ChatMessage msg : messages) {
            switch (msg.getRole()) {
                case "user" -> sb.append("用户: ").append(msg.getContent()).append("\n");
                case "assistant" -> sb.append("助手: ").append(msg.getContent()).append("\n");
                case "tool_call" -> sb.append("[调用工具: ").append(msg.getToolName()).append("]\n");
                case "tool_response" -> sb.append("[工具结果: ").append(msg.getContent()).append("]\n");
            }
        }
        return sb.toString();
    }

    /**
     * 清除会话的短期记忆（会话结束时调用）
     */
    public void clear(Long sessionId) {
        sessionMessages.remove(sessionId);
        log.info("短期记忆已清除: sessionId={}", sessionId);
    }

    /**
     * 提取关键对话片段（用于写入长期记忆）
     * 策略：筛选包含情绪变化、风险事件、重要结论的消息
     */
    public List<ChatMessage> extractKeyFragments(Long sessionId) {
        List<ChatMessage> all = getMessages(sessionId);
        List<ChatMessage> keyFragments = new ArrayList<>();

        for (ChatMessage msg : all) {
            if (msg.getRole().equals("assistant") && isKeyMessage(msg.getContent())) {
                keyFragments.add(msg);
            }
        }
        // 至少保留最后一条，避免空结果
        if (keyFragments.isEmpty() && !all.isEmpty()) {
            keyFragments.add(all.get(all.size() - 1));
        }
        return keyFragments;
    }

    private boolean isKeyMessage(String content) {
        if (content == null) return false;
        String[] keywords = {"焦虑", "抑郁", "失眠", "恐慌", "自杀", "绝望",
                "建议", "放松", "咨询", "风险", "情绪", "心理"};
        for (String kw : keywords) {
            if (content.contains(kw)) return true;
        }
        return false;
    }
}
