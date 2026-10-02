package com.dss.ai.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * AI 配置(从 application-ai-key.yml 读)。
 * key 不进 Git,放 application-ai-key.yml(已 gitignore)。
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "ai")
public class AiProperties {

    private DeepSeek deepseek = new DeepSeek();
    private Ollama ollama = new Ollama();

    @Data
    public static class DeepSeek {
        private String apiKey;
        private String baseUrl = "https://api.deepseek.com";
        private String model = "deepseek-chat";
    }

    @Data
    public static class Ollama {
        private String baseUrl = "http://localhost:11434";
        private String embeddingModel = "bge-m3";
    }
}
