package com.mindguard;

import com.mindguard.memory.ShortTermMemory;
import com.mindguard.model.ChatMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MemoryServiceTest {

    private ShortTermMemory shortTermMemory;

    @BeforeEach
    void setUp() {
        // 需要 mock ChatMessageRepository
    }

    @Test
    void testSlidingWindow() {
        // 验证滑动窗口：超出 maxMessages 时移除最早的消息
    }

    @Test
    void testKeyFragmentExtraction() {
        // 验证关键片段提取：包含焦虑/抑郁等关键词的消息被选中
    }
}