package com.dss.ai.tool;

import com.dss.ai.dto.ToolDef;

import java.util.Map;

/**
 * AI 工具接口:每个工具是一个可被 LLM 调用的函数。
 * 类似 MCP 的 tool 概念——把后端能力暴露给 LLM。
 */
public interface AiTool {

    /** 工具名(给 LLM 看,需唯一) */
    String name();

    /** 工具描述(告诉 LLM 何时用这个工具) */
    String description();

    /** 参数 JSON Schema(告诉 LLM 怎么传参) */
    Map<String, Object> parameters();

    /** 执行工具,返回结果字符串(给 LLM 看) */
    String execute(Map<String, Object> args);

    /** 转成 OpenAI Function Calling 格式 */
    default ToolDef toToolDef() {
        return ToolDef.function(name(), description(), parameters());
    }
}
