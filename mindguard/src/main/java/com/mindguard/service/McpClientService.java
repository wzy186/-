package com.mindguard.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

/**
 * MCP 服务层：封装所有通过 MCP 协议对接的外部服务。
 * 包括：邮件通知、Excel 记录写入、辅导员预约
 * 实际生产中这些会通过 MCP Server 暴露，此处为本地实现
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class McpClientService {

    private final JavaMailSender mailSender;

    @Value("${mindguard.risk.high-risk-email-enabled:true}")
    private boolean emailEnabled;

    @Value("${mindguard.risk.high-risk-email-to:}")
    private String highRiskEmailTo;

    @Value("${mindguard.excel.path:./data/psychological_records.xlsx}")
    private String excelPath;

    // ===== 邮件通知（MCP: sendEmail） =====

    /**
     * 发送高风险告警邮件给负责人
     */
    public boolean sendHighRiskAlert(Long userId, String username,
                                     String riskLevel, String summary) {
        if (!emailEnabled) {
            log.info("邮件通知已禁用，跳过发送");
            return false;
        }

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(highRiskEmailTo.split(","));
            message.setSubject("[MindGuard 告警] 检测到" + riskLevel + "学生");
            message.setText(String.format(
                    "检测到高风险学生，请及时关注：\n\n" +
                    "学生ID: %d\n学生姓名: %s\n风险等级: %s\n\n" +
                    "对话摘要:\n%s\n\n" +
                    "此邮件由 MindGuard AI 自动发送，请勿直接回复。",
                    userId, username, riskLevel, summary
            ));
            mailSender.send(message);
            log.info("高风险告警邮件已发送: userId={}, riskLevel={}", userId, riskLevel);
            return true;
        } catch (Exception e) {
            log.error("邮件发送失败: userId={}", userId, e);
            return false;
        }
    }

    // ===== Excel 记录写入（MCP: writeExcel） =====

    /**
     * 将心理状态记录写入 Excel
     */
    public boolean writePsychologicalRecord(Long userId, String username,
                                            String emotionLabel, String riskLevel,
                                            String summary) {
        try {
            org.apache.poi.ss.usermodel.Workbook workbook;
            java.io.File file = new java.io.File(excelPath);
            org.apache.poi.ss.usermodel.Sheet sheet;

            if (file.exists()) {
                workbook = new org.apache.poi.xssf.usermodel.XSSFWorkbook(new java.io.FileInputStream(file));
                sheet = workbook.getSheetAt(0);
            } else {
                file.getParentFile().mkdirs();
                workbook = new org.apache.poi.xssf.usermodel.XSSFWorkbook();
                sheet = workbook.createSheet("心理监护记录");
                // 表头
                org.apache.poi.ss.usermodel.Row header = sheet.createRow(0);
                String[] headers = {"时间", "学生ID", "学生姓名", "情绪标签", "风险等级", "对话摘要"};
                for (int i = 0; i < headers.length; i++) {
                    header.createCell(i).setCellValue(headers[i]);
                }
            }

            int rowNum = sheet.getLastRowNum() + 1;
            org.apache.poi.ss.usermodel.Row row = sheet.createRow(rowNum);
            row.createCell(0).setCellValue(java.time.LocalDateTime.now().toString());
            row.createCell(1).setCellValue(userId);
            row.createCell(2).setCellValue(username);
            row.createCell(3).setCellValue(emotionLabel);
            row.createCell(4).setCellValue(riskLevel);
            row.createCell(5).setCellValue(summary);

            // 原子保存
            String tmpPath = excelPath + ".tmp";
            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(tmpPath)) {
                workbook.write(fos);
            }
            new java.io.File(tmpPath).renameTo(file);
            workbook.close();

            log.info("心理记录已写入Excel: userId={}, emotion={}, risk={}",
                    userId, emotionLabel, riskLevel);
            return true;
        } catch (Exception e) {
            log.error("Excel写入失败: userId={}", userId, e);
            return false;
        }
    }

    // ===== 辅导员预约（MCP: bookCounselor） =====

    /**
     * 预约辅导员
     * 实际生产中对接预约系统 API，此处为模拟实现
     */
    public String bookCounselor(Long userId, String username, String reason) {
        // 模拟预约逻辑
        String bookingId = "BK" + System.currentTimeMillis();
        log.info("辅导员预约成功: userId={}, bookingId={}, reason={}",
                userId, bookingId, reason);
        return String.format("预约成功！预约编号：%s，辅导员将在24小时内联系您。", bookingId);
    }
}
