package com.mindguard.tools;

import com.mindguard.service.McpClientService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 邮件通知工具：高风险学生自动发送告警邮件给负责人
 * 通过 MCP 协议对接邮件服务
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class SendEmailTool {

    private final McpClientService mcpClientService;

    /**
     * 发送高风险告警邮件
     * @param userId   学生 ID
     * @param username 学生姓名
     * @param riskLevel 风险等级
     * @param summary   对话摘要
     * @return 发送结果
     */
    public String sendAlert(Long userId, String username,
                            String riskLevel, String summary) {
        log.info("[Tool] SendEmail 被调用: userId={}, riskLevel={}", userId, riskLevel);
        boolean success = mcpClientService.sendHighRiskAlert(userId, username, riskLevel, summary);
        return success ? "告警邮件已成功发送给相关负责人。" : "邮件发送失败，请检查邮件服务配置。";
    }
}
