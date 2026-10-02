package com.dss.ai.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * 工具定义(OpenAI Function Calling 格式)。
 * 暴露给 LLM,让它知道有哪些工具可调。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ToolDef {
    private String type = "function";
    private Function function;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Function {
        private String name;
        private String description;
        private Map<String, Object> parameters; // JSON Schema
    }

    public static ToolDef function(String name, String desc, Map<String, Object> params) {
        return ToolDef.builder()
                .type("function")
                .function(Function.builder().name(name).description(desc).parameters(params).build())
                .build();
    }
}
