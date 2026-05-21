package com.mindguard.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

/**
 * 知识库文档服务：自动构建知识库，将文档向量化存入 Chroma。
 * - 支持管理员上传文档（PDF、TXT、Markdown）
 * - 自动切分为语义完整的 chunk
 * - 通过 Chroma 向量存储支持语义检索
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DocumentService {

    private final VectorStore vectorStore;

    @Value("${mindguard.knowledge.base-dir:./knowledge/}")
    private String knowledgeBaseDir;

    @Value("${mindguard.memory.long-term.chunk-size:300}")
    private int chunkSize;

    @Value("${mindguard.memory.long-term.chunk-overlap:50}")
    private int chunkOverlap;

    /**
     * 自动扫描知识库目录，将所有文档导入 Chroma
     */
    public int autoIngestKnowledgeBase() {
        File dir = new File(knowledgeBaseDir);
        if (!dir.exists() || !dir.isDirectory()) {
            log.warn("知识库目录不存在: {}", knowledgeBaseDir);
            return 0;
        }

        int totalChunks = 0;
        File[] files = dir.listFiles((d, name) ->
                name.endsWith(".txt") || name.endsWith(".md") ||
                name.endsWith(".pdf") || name.endsWith(".docx"));

        if (files == null) return 0;

        for (File file : files) {
            try {
                int chunks = ingestDocument(file);
                totalChunks += chunks;
                log.info("知识库文档已导入: {}, chunks={}", file.getName(), chunks);
            } catch (Exception e) {
                log.error("导入文档失败: {}", file.getName(), e);
            }
        }
        return totalChunks;
    }

    /**
     * 导入单个文档到 Chroma
     */
    public int ingestDocument(File file) throws Exception {
        String content = Files.readString(file.toPath());
        return ingestText(content, file.getName(), "knowledge_base");
    }

    /**
     * 导入文本内容到 Chroma（通用方法）
     */
    public int ingestText(String content, String docName, String category) {
        TokenTextSplitter splitter = new TokenTextSplitter(chunkSize, chunkOverlap, 5, 10000, true);

        Document doc = new Document(content, Map.of(
                "source", docName,
                "category", category,
                "ingestedAt", String.valueOf(System.currentTimeMillis()),
                "type", "knowledge"
        ));

        List<Document> chunks = splitter.split(List.of(doc));
        vectorStore.add(chunks);
        return chunks.size();
    }

    /**
     * RAG 检索：根据查询语义检索知识库
     */
    public List<String> searchKnowledge(String query, int topK) {
        List<Document> results = vectorStore.similaritySearch(SearchRequest.builder().query(query).topK(topK).build());

        return results.stream()
                .filter(doc -> "knowledge".equals(doc.getMetadata().get("type")))
                .limit(topK)
                .map(doc -> String.format("[知识库 | 来源:%s] %s",
                        doc.getMetadata().getOrDefault("source", "unknown"),
                        doc.getContent()))
                .toList();
    }
}
