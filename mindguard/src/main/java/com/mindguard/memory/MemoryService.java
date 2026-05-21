package com.mindguard.memory;

import com.mindguard.model.ChatMessage;
import com.mindguard.model.ChatSession;
import com.mindguard.model.EmotionLabel;
import com.mindguard.model.RiskLevel;
import com.mindguard.repository.ChatSessionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 记忆服务：协调短期记忆与长期记忆的协同工作。
 * - 会话进行中：短期记忆维持对话连贯性
 * - 会话结束时：提取关键片段 → 写入长期记忆 → 清除短期记忆
 * - 跨会话查询：从长期记忆语义检索历史对话
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MemoryService {

    private final ShortTermMemory shortTermMemory;
    private final LongTermMemory longTermMemory;
    private final ChatSessionRepository chatSessionRepository;

    /**
     * 记录一条消息到短期记忆
     */
    public void recordMessage(Long sessionId, Long userId, String role,
                              String content, String toolName, String toolArgs) {
        shortTermMemory.addMessage(sessionId, userId, role, content, toolName, toolArgs);
    }

    /**
     * 获取当前会话的完整上下文（短期记忆格式化）
     */
    public String getCurrentContext(Long sessionId) {
        return shortTermMemory.formatAsContext(sessionId);
    }

    /**
     * 查询长期记忆（跨会话历史检索）
     */
    public List<String> queryHistory(Long userId, String query) {
        return longTermMemory.searchHistory(userId, query);
    }

    /**
     * 会话结束时的记忆处理：
     * 1. 从短期记忆提取关键片段
     * 2. 写入长期记忆（Chroma 向量库）
     * 3. 更新会话记录（风险等级、摘要）
     * 4. 清除短期记忆
     */
    public void onSessionEnd(Long sessionId, Long userId,
                             RiskLevel riskLevel, EmotionLabel emotionLabel,
                             String summary) {
        // 1. 提取关键片段
        List<ChatMessage> keyFragments = shortTermMemory.extractKeyFragments(sessionId);
        List<String> fragmentTexts = keyFragments.stream()
                .map(ChatMessage::getContent)
                .collect(Collectors.toList());

        // 2. 写入长期记忆
        longTermMemory.storeKeyFragments(userId, sessionId, fragmentTexts,
                emotionLabel != null ? emotionLabel.name() : null,
                riskLevel != null ? riskLevel.name() : null);

        // 3. 更新会话记录
        chatSessionRepository.findById(sessionId).ifPresent(session -> {
            session.setEndTime(LocalDateTime.now());
            session.setFinalRiskLevel(riskLevel);
            session.setSummary(summary);
            session.setArchived(true);
            chatSessionRepository.save(session);
        });

        // 4. 清除短期记忆
        shortTermMemory.clear(sessionId);

        log.info("会话记忆处理完成: sessionId={}, riskLevel={}, fragments={}",
                sessionId, riskLevel, fragmentTexts.size());
    }

    /**
     * 构建增强 Prompt：System Prompt + 长期记忆 + 短期记忆 + 当前输入
     */
    public String buildEnhancedPrompt(Long userId, Long sessionId,
                                      String systemPrompt, String currentInput) {
        StringBuilder prompt = new StringBuilder();

        // 1. System Prompt（含角色约束）
        prompt.append(systemPrompt).append("\n\n");

        // 2. 长期记忆注入
        List<String> history = longTermMemory.searchHistory(userId, currentInput);
        if (!history.isEmpty()) {
            prompt.append("=== 该用户的历史关键信息 ===\n");
            for (String h : history) {
                prompt.append(h).append("\n");
            }
            prompt.append("\n");
        }

        // 3. 短期记忆（当前会话上下文）
        String currentContext = shortTermMemory.formatAsContext(sessionId);
        if (!currentContext.isEmpty()) {
            prompt.append("=== 当前会话上下文 ===\n");
            prompt.append(currentContext).append("\n");
        }

        return prompt.toString();
    }
}
