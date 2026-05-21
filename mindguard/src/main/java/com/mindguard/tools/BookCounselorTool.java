package com.mindguard.tools;

import com.mindguard.service.McpClientService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 辅导员预约工具：为学生预约线下心理咨询
 * 通过 MCP 协议对接预约系统
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class BookCounselorTool {

    private final McpClientService mcpClientService;

    /**
     * 预约辅导员
     * @param userId   学生 ID
     * @param username 学生姓名
     * @param reason   预约原因
     * @return 预约结果
     */
    public String book(Long userId, String username, String reason) {
        log.info("[Tool] BookCounselor 被调用: userId={}, reason={}", userId, reason);
        return mcpClientService.bookCounselor(userId, username, reason);
    }
}
