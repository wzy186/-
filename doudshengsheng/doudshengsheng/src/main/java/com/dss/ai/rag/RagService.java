package com.dss.ai.rag;

import com.dss.ai.client.EmbeddingClient;
import com.dss.ai.client.LlmClient;
import com.dss.ai.dto.ChatMessage;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.entity.Blog;
import com.dss.entity.Shop;
import com.dss.mapper.BlogMapper;
import com.dss.mapper.ShopMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * RAG 语义问答服务。
 * <p>
 * 流程:
 * 1. 启动时把商铺/笔记文本向量化,存内存向量库
 * 2. 用户问题向量化 → 余弦相似度检索 top-k → 拼进 prompt → LLM 生成
 * <p>
 * 注:演示用内存向量库。生产应换 RediSearch/Milvus(支持持久化 + 海量检索)。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RagService {

    private final EmbeddingClient embeddingClient;
    private final LlmClient llmClient;
    private final ShopMapper shopMapper;
    private final BlogMapper blogMapper;

    /** 向量库:docId -> 向量 */
    private final Map<String, float[]> vectorStore = new ConcurrentHashMap<>();
    /** 文档:docId -> 文本 */
    private final Map<String, String> docStore = new ConcurrentHashMap<>();

    /**
     * 重建索引:把商铺和笔记向量化。
     */
    public int reindex() {
        vectorStore.clear();
        docStore.clear();
        int count = 0;

        // 商铺
        List<Shop> shops = shopMapper.selectList(new LambdaQueryWrapper<Shop>().last("limit 100"));
        for (Shop s : shops) {
            String text = String.format("商铺:%s。类型%d。地址:%s。均价%d元/人。评分%.1f。销量%d。",
                    s.getName(), s.getTypeId(), s.getAddress(),
                    s.getAvgPrice() == null ? 0 : s.getAvgPrice() / 100,
                    s.getScore() == null ? 0 : s.getScore() / 20.0,
                    s.getSold() == null ? 0 : s.getSold());
            String docId = "shop:" + s.getId();
            try {
                vectorStore.put(docId, embeddingClient.embed(text));
                docStore.put(docId, text);
                count++;
            } catch (Exception e) {
                log.warn("商铺 {} 向量化失败: {}", s.getId(), e.getMessage());
            }
        }

        // 笔记
        List<Blog> blogs = blogMapper.selectList(new LambdaQueryWrapper<Blog>().last("limit 100"));
        for (Blog b : blogs) {
            String text = String.format("探店笔记:%s。内容:%s。点赞%d。",
                    b.getTitle(), b.getContent() == null ? "" : b.getContent(),
                    b.getLiked() == null ? 0 : b.getLiked());
            String docId = "blog:" + b.getId();
            try {
                vectorStore.put(docId, embeddingClient.embed(text));
                docStore.put(docId, text);
                count++;
            } catch (Exception e) {
                log.warn("笔记 {} 向量化失败: {}", b.getId(), e.getMessage());
            }
        }
        log.info("RAG 索引完成,共 {} 条文档", count);
        return count;
    }

    /**
     * RAG 问答。
     * 优先向量检索;embedding 服务不可用时降级为关键词检索。
     */
    public String ask(String query) {
        List<String> topDocs;

        // 尝试向量检索
        if (!vectorStore.isEmpty()) {
            try {
                float[] qVec = embeddingClient.embed(query);
                List<Map.Entry<String, Double>> top = new ArrayList<>();
                for (Map.Entry<String, float[]> e : vectorStore.entrySet()) {
                    top.add(new AbstractMap.SimpleEntry<>(e.getKey(), EmbeddingClient.cosine(qVec, e.getValue())));
                }
                top.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
                topDocs = top.stream().limit(5).map(e -> docStore.get(e.getKey())).toList();
            } catch (Exception e) {
                log.warn("向量检索失败,降级关键词: {}", e.getMessage());
                topDocs = keywordSearch(query);
            }
        } else {
            // 向量库为空(embedding 服务没就绪),直接关键词检索
            topDocs = keywordSearch(query);
        }

        if (topDocs.isEmpty()) {
            topDocs = List.of("（未检索到相关商铺/笔记）");
        }

        // 拼 prompt
        StringBuilder context = new StringBuilder();
        for (String doc : topDocs) {
            context.append("- ").append(doc).append("\n");
        }
        String prompt = """
            你是兜省省的省钱助手。根据以下检索到的商铺/笔记信息回答用户问题。
            如果信息不足,如实说明。不要编造未在资料中出现的商铺。

            【参考资料】
            %s

            【用户问题】
            %s
            """.formatted(context, query);

        LlmClient.ChatResult result = llmClient.chat(List.of(
                ChatMessage.system("你是省钱助手,基于资料如实回答。"),
                ChatMessage.user(prompt)
        ), null);
        return result.content();
    }

    /**
     * 关键词检索(降级方案):从商铺和笔记里按关键词匹配。
     */
    private List<String> keywordSearch(String query) {
        List<String> docs = new ArrayList<>();
        // 商铺:名/地址匹配
        List<Shop> shops = shopMapper.selectList(new LambdaQueryWrapper<Shop>()
                .like(Shop::getName, query).or().like(Shop::getAddress, query).last("limit 5"));
        for (Shop s : shops) {
            docs.add(String.format("商铺:%s。地址:%s。均价%d元/人。评分%.1f。",
                    s.getName(), s.getAddress(),
                    s.getAvgPrice() == null ? 0 : s.getAvgPrice() / 100,
                    s.getScore() == null ? 0 : s.getScore() / 20.0));
        }
        // 笔记:标题/内容匹配
        List<Blog> blogs = blogMapper.selectList(new LambdaQueryWrapper<Blog>()
                .like(Blog::getTitle, query).or().like(Blog::getContent, query).last("limit 5"));
        for (Blog b : blogs) {
            docs.add(String.format("笔记:%s。内容:%s", b.getTitle(),
                    b.getContent() == null ? "" : (b.getContent().length() > 60 ? b.getContent().substring(0, 60) : b.getContent())));
        }
        return docs;
    }

    public int docCount() {
        return vectorStore.size();
    }
}
