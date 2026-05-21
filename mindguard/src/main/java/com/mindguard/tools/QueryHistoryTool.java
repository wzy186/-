package com.mindguard.tools;

import com.mindguard.memory.LongTermMemory;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 历史对话查询工具：从 Chroma 长期记忆中检索用户的历史对话
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class QueryHistoryTool {

    private final LongTermMemory longTermMemory;

    /**
     * 查询用户历史对话
     * @param userId 用户 ID
     * @param query  当前输入（用于语义匹配）
     * @return 历史对话字符串
     */
    public String query(Long userId, String query) {
        log.info("[Tool] QueryHistory 被调用: userId={}, query={}", userId, query);
        List<String> history = longTermMemory.searchHistory(userId, query);
        if (history.isEmpty()) {
            return "未找到相关历史对话记录。";
        }
        return String.join("\n", history);
    }
}
