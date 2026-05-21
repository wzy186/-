package com.mindguard;

import com.mindguard.agent.MindGuardAgent;
import com.mindguard.memory.MemoryService;
import com.mindguard.tools.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class MindGuardAgentTest {

    @Mock private MemoryService memoryService;
    @Mock private RagSearchTool ragSearchTool;
    @Mock private QueryHistoryTool queryHistoryTool;
    @Mock private SendEmailTool sendEmailTool;
    @Mock private WriteExcelTool writeExcelTool;
    @Mock private BookCounselorTool bookCounselorTool;

    private MindGuardAgent agent;

    @BeforeEach
    void setUp() {
        // 实际测试需要注入真实的 ChatModel
        // agent = new MindGuardAgent(...);
    }

    @Test
    void testToolRegistryInit() {
        // 验证工具注册表包含所有5个工具
        // agent.init();
        // assertEquals(5, agent.toolRegistry.size());
    }
}