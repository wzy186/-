package com.mindguard.memory;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 长期记忆：基于向量数据库的跨会话记忆系统。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class LongTermMemory {

    private final VectorStore vectorStore;

    @Value("${mindguard.memory.long-term.top-k:5}")
    private int topK;

    @Value("${mindguard.memory.long-term.chunk-size:300}")
    private int chunkSize;

    @Value("${mindguard.memory.long-term.chunk-overlap:50}")
    private int chunkOverlap;

    private static final String COLLECTION_NAME = "mindguard_long_term_memory";

    /**
     * 存储关键对话片段到长期记忆
     *
     * @param userId   用户 ID
     * @param sessionId 会话 ID
     * @param fragments 关键对话片段列表
     * @param emotionLabel 情绪标签
     * @param riskLevel 风险等级
     */
    public void storeKeyFragments(Long userId, Long sessionId,
                                   List<String> fragments,
                                   String emotionLabel, String riskLevel) {
        if (fragments == null || fragments.isEmpty()) {
            log.warn("无关键片段可存储: userId={}", userId);
            return;
        }

        // 合并片段为完整文本
        String fullText = String.join("\n", fragments);

        // 使用 TokenTextSplitter 进行语义切分
        TokenTextSplitter splitter = new TokenTextSplitter(chunkSize, chunkOverlap, 5, 10000, true);

        // 构建带元数据的 Document
        Document doc = new Document(fullText, Map.of(
                "userId", String.valueOf(userId),
                "sessionId", String.valueOf(sessionId),
                "emotionLabel", emotionLabel != null ? emotionLabel : "NORMAL",
                "riskLevel", riskLevel != null ? riskLevel : "LOW",
                "timestamp", String.valueOf(System.currentTimeMillis()),
                "type", "conversation_fragment"
        ));

        // 切分并存储
        List<Document> chunks = splitter.split(List.of(doc));
        vectorStore.add(chunks);

        log.info("长期记忆已存储: userId={}, chunks={}", userId, chunks.size());
    }

    /**
     * 语义检索历史对话
     *
     * @param userId 用户 ID
     * @param query  查询文本（当前用户输入）
     * @return 相似度最高的 Top-K 条历史对话
     */
    public List<String> searchHistory(Long userId, String query) {
        // 构建带过滤条件的检索
        List<Document> results = vectorStore.similaritySearch(SearchRequest.builder().query(query).topK(topK).build());

        // 按 userId 过滤（Chroma 的 metadata filter）
        List<String> history = results.stream()
                .filter(doc -> String.valueOf(userId).equals(doc.getMetadata().get("userId")))
                .limit(topK)
                .map(doc -> {
                    String emotion = (String) doc.getMetadata().getOrDefault("emotionLabel", "");
                    String risk = (String) doc.getMetadata().getOrDefault("riskLevel", "");
                    String ts = (String) doc.getMetadata().getOrDefault("timestamp", "");
                    return String.format("[历史记忆 | 情绪:%s | 风险:%s | 时间:%s] %s",
                            emotion, risk, ts, doc.getContent());
                })
                .collect(Collectors.toList());

        log.info("长期记忆检索: userId={}, query={}, hits={}", userId, query, history.size());
        return history;
    }

    /**
     * 获取用户的长期记忆概要（用于管理员查看）
     */
    public List<Map<String, String>> getUserMemoryOverview(Long userId) {
        List<Document> allDocs = vectorStore.similaritySearch(SearchRequest.builder().query("").topK(100).build());
        return allDocs.stream()
                .filter(doc -> String.valueOf(userId).equals(doc.getMetadata().get("userId")))
                .map(doc -> {
                    Map<String, String> overview = new HashMap<>();
                    overview.put("content", doc.getContent());
                    overview.put("emotionLabel", (String) doc.getMetadata().getOrDefault("emotionLabel", ""));
                    overview.put("riskLevel", (String) doc.getMetadata().getOrDefault("riskLevel", ""));
                    overview.put("timestamp", (String) doc.getMetadata().getOrDefault("timestamp", ""));
                    return overview;
                })
                .collect(Collectors.toList());
    }
}
