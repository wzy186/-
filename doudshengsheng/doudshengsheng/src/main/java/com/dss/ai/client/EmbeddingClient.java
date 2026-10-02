package com.dss.ai.client;

import com.dss.ai.config.AiProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Embedding 客户端:调本地 ollama 把文本转向量。
 * 用 bge-m3 模型(中文语义检索效果好),本地跑不耗 API 额度。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class EmbeddingClient {

    private final AiProperties props;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();

    /**
     * 把一段文本转向量(float 数组)。
     */
    @SuppressWarnings("unchecked")
    public float[] embed(String text) {
        if (text == null || text.isBlank()) {
            return new float[0];
        }
        String t = text.length() > 2000 ? text.substring(0, 2000) : text;

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", props.getOllama().getEmbeddingModel());
        body.put("input", t);

        try {
            String json = objectMapper.writeValueAsString(body);
            // 优先新版接口 /api/embed,失败回退旧版 /api/embeddings
            String[] endpoints = {"/api/embed", "/api/embeddings"};
            Exception lastErr = null;
            for (String ep : endpoints) {
                Request req = new Request.Builder()
                        .url(props.getOllama().getBaseUrl() + ep)
                        .post(RequestBody.create(json, MediaType.parse("application/json")))
                        .build();
                try (Response resp = httpClient.newCall(req).execute()) {
                    String respBody = resp.body() == null ? "" : resp.body().string();
                    if (!resp.isSuccessful()) {
                        lastErr = new RuntimeException("embedding " + resp.code() + ": " + respBody);
                        continue;
                    }
                    Map<String, Object> map = objectMapper.readValue(respBody, Map.class);
                    List<Number> vec = (List<Number>) map.get("embedding"); // 旧接口
                    if (vec == null || vec.isEmpty()) {
                        // 新接口返回 embeddings(数组嵌套)
                        List<List<Number>> vecs = (List<List<Number>>) map.get("embeddings");
                        if (vecs != null && !vecs.isEmpty()) vec = vecs.get(0);
                    }
                    if (vec != null && !vec.isEmpty()) {
                        float[] result = new float[vec.size()];
                        for (int i = 0; i < vec.size(); i++) {
                            result[i] = vec.get(i).floatValue();
                        }
                        return result;
                    }
                } catch (Exception e) {
                    lastErr = e;
                }
            }
            throw new RuntimeException("embedding 为空" + (lastErr == null ? "" : ": " + lastErr.getMessage()));
        } catch (Exception e) {
            log.error("Embedding 失败: {}", e.getMessage());
            throw new RuntimeException("Embedding 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 余弦相似度(向量检索用)。
     */
    public static double cosine(float[] a, float[] b) {
        if (a.length != b.length || a.length == 0) return 0;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
