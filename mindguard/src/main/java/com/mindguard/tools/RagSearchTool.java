package com.mindguard.tools;

import com.mindguard.service.DocumentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.stereotype.Component;

/**
 * RAG 检索工具：Agent 自主决策是否需要检索知识库
 * 通过 SpringAI Function Calling 注册，LLM 自主判断调用时机
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RagSearchTool {

    private final DocumentService documentService;

    /**
     * 检索知识库
     * @param query 检索查询
     * @return 检索结果字符串
     */
    public String search(String query) {
        log.info("[Tool] RagSearch 被调用: query={}", query);
        var results = documentService.searchKnowledge(query, 5);
        if (results.isEmpty()) {
            return "知识库中未找到相关内容。";
        }
        return String.join("\n\n", results);
    }
}
