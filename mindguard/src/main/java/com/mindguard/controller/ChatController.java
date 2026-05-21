package com.mindguard.controller;

import com.mindguard.model.ChatMessage;
import com.mindguard.model.ChatSession;
import com.mindguard.model.PsychologicalRecord;
import com.mindguard.repository.ChatMessageRepository;
import com.mindguard.repository.ChatSessionRepository;
import com.mindguard.repository.PsychologicalRecordRepository;
import com.mindguard.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;
    private final ChatSessionRepository chatSessionRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final PsychologicalRecordRepository psychologicalRecordRepository;

    private Long getCurrentUserId() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return (Long) auth.getDetails();
    }

    @PostMapping("/send")
    public ResponseEntity<?> chat(@RequestBody Map<String, String> body) {
        Long userId = getCurrentUserId();
        String userInput = body.get("message");

        if (userInput == null || userInput.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "消息不能为空"));
        }

        String reply = chatService.chat(userId, userInput);
        return ResponseEntity.ok(Map.of("reply", reply));
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> chatStream(@RequestBody Map<String, String> body) {
        Long userId = getCurrentUserId();
        String userInput = body.get("message");
        return chatService.chatStream(userId, userInput);
    }

    @PostMapping("/end-session")
    public ResponseEntity<?> endSession(@RequestBody Map<String, String> body) {
        Long userId = getCurrentUserId();
        chatService.endSession(userId,
                com.mindguard.model.RiskLevel.valueOf(body.getOrDefault("riskLevel", "LOW")),
                com.mindguard.model.EmotionLabel.valueOf(body.getOrDefault("emotionLabel", "NORMAL")),
                body.getOrDefault("summary", ""));
        return ResponseEntity.ok(Map.of("message", "会话已结束"));
    }

    /** 查询当前用户的会话列表 */
    @GetMapping("/sessions")
    public ResponseEntity<?> getMySessions() {
        Long userId = getCurrentUserId();
        List<ChatSession> sessions = chatSessionRepository.findByUserIdOrderByStartTimeDesc(userId);
        return ResponseEntity.ok(sessions);
    }

    /** 查询指定会话的消息记录 */
    @GetMapping("/sessions/{sessionId}/messages")
    public ResponseEntity<?> getSessionMessages(@PathVariable Long sessionId) {
        Long userId = getCurrentUserId();
        ChatSession session = chatSessionRepository.findById(sessionId).orElse(null);
        if (session == null || !session.getUserId().equals(userId)) {
            return ResponseEntity.status(403).body(Map.of("error", "无权访问"));
        }
        List<ChatMessage> messages = chatMessageRepository.findBySessionIdOrderByCreatedAtAsc(sessionId);
        return ResponseEntity.ok(messages);
    }

    /** 查询当前用户的心理监护记录 */
    @GetMapping("/records")
    public ResponseEntity<?> getMyRecords() {
        Long userId = getCurrentUserId();
        List<PsychologicalRecord> records = psychologicalRecordRepository.findByUserIdOrderByRecordedAtDesc(userId);
        return ResponseEntity.ok(records);
    }
}
