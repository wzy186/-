package com.mindguard.service;

import com.mindguard.agent.MindGuardAgent;
import com.mindguard.memory.MemoryService;
import com.mindguard.model.*;
import com.mindguard.repository.ChatSessionRepository;
import com.mindguard.repository.PsychologicalRecordRepository;
import com.mindguard.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.time.LocalDateTime;

/**
 * 聊天服务：串联 Agent + 记忆系统 + 会话管理
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    private final MindGuardAgent mindGuardAgent;
    private final MemoryService memoryService;
    private final ChatSessionRepository chatSessionRepository;
    private final PsychologicalRecordRepository psychologicalRecordRepository;
    private final UserRepository userRepository;

    /**
     * 处理用户消息（同步接口）
     */
    public String chat(Long userId, String userInput) {
        User user = userRepository.findById(userId).orElseThrow();
        String username = user.getRealName() != null ? user.getRealName() : user.getUsername();
        String role = user.getRole().name();

        // 获取或创建活跃会话
        ChatSession session = getOrCreateSession(userId);

        // Agent 处理
        String reply = mindGuardAgent.process(userId, username, session.getId(), role, userInput);

        return reply;
    }

    /**
     * 流式聊天接口（SSE）
     */
    public Flux<String> chatStream(Long userId, String userInput) {
        // 先同步获取 Agent 完整回复，再分段流式输出
        return Flux.create(sink -> {
            try {
                String reply = chat(userId, userInput);
                // 模拟流式输出：按句号分段
                String[] sentences = reply.split("(?<=[。！？.!?])");
                for (String sentence : sentences) {
                    sink.next(sentence);
                }
                sink.complete();
            } catch (Exception e) {
                sink.error(e);
            }
        });
    }

    /**
     * 结束会话：触发记忆归档
     */
    public void endSession(Long userId, RiskLevel riskLevel, EmotionLabel emotionLabel, String summary) {
        ChatSession session = getOrCreateSession(userId);
        memoryService.onSessionEnd(session.getId(), userId, riskLevel, emotionLabel, summary);

        // 保存心理记录
        PsychologicalRecord record = PsychologicalRecord.builder()
                .userId(userId)
                .sessionId(session.getId())
                .emotionLabel(emotionLabel)
                .riskLevel(riskLevel)
                .conversationSummary(summary)
                .build();
        psychologicalRecordRepository.save(record);

        log.info("会话已结束并归档: userId={}, sessionId={}, riskLevel={}",
                userId, session.getId(), riskLevel);
    }

    /**
     * 获取或创建活跃会话
     */
    private ChatSession getOrCreateSession(Long userId) {
        return chatSessionRepository
                .findTopByUserIdAndEndTimeIsNullOrderByStartTimeDesc(userId)
                .orElseGet(() -> chatSessionRepository.save(
                        ChatSession.builder()
                                .userId(userId)
                                .startTime(LocalDateTime.now())
                                .build()
                ));
    }
}
