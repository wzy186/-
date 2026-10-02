package com.dss.ai.tool;

import com.dss.ai.dto.ToolDef;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 工具注册中心:收集所有 AiTool,供 Agent 调用。
 * 相当于 MCP Server 的工具表——LLM 从这里知道能调什么工具。
 */
@Slf4j
@Component
public class ToolRegistry {

    private final Map<String, AiTool> tools = new LinkedHashMap<>();

    public ToolRegistry(List<AiTool> toolList) {
        for (AiTool t : toolList) {
            tools.put(t.name(), t);
            log.info("注册 AI 工具: {}", t.name());
        }
    }

    /** 获取所有工具定义(给 LLM) */
    public List<ToolDef> allToolDefs() {
        return tools.values().stream().map(AiTool::toToolDef).toList();
    }

    /** 按名执行工具 */
    public String execute(String name, Map<String, Object> args) {
        AiTool t = tools.get(name);
        if (t == null) {
            return "工具不存在: " + name;
        }
        try {
            return t.execute(args == null ? Map.of() : args);
        } catch (Exception e) {
            log.error("工具 {} 执行失败: {}", name, e.getMessage());
            return "工具执行失败: " + e.getMessage();
        }
    }

    public Set<String> names() {
        return tools.keySet();
    }
}
