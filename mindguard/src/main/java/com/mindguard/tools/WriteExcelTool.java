package com.mindguard.tools;

import com.mindguard.service.McpClientService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * Excel 记录写入工具：将心理状态自动写入 Excel
 * 通过 MCP 协议对接 Excel 服务
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class WriteExcelTool {

    private final McpClientService mcpClientService;

    /**
     * 写入心理状态记录到 Excel
     * @param userId      学生 ID
     * @param username    学生姓名
     * @param emotionLabel 情绪标签
     * @param riskLevel   风险等级
     * @param summary     对话摘要
     * @return 写入结果
     */
    public String writeRecord(Long userId, String username,
                              String emotionLabel, String riskLevel,
                              String summary) {
        log.info("[Tool] WriteExcel 被调用: userId={}, emotion={}, risk={}",
                userId, emotionLabel, riskLevel);
        boolean success = mcpClientService.writePsychologicalRecord(
                userId, username, emotionLabel, riskLevel, summary);
        return success ? "心理状态记录已成功写入Excel。" : "Excel写入失败。";
    }
}
