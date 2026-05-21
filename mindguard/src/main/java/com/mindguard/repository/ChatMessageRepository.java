package com.mindguard.repository;

import com.mindguard.model.ChatMessage;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    List<ChatMessage> findBySessionIdOrderByCreatedAtAsc(Long sessionId);
    List<ChatMessage> findByUserIdOrderByCreatedAtDesc(Long userId);
    List<ChatMessage> findTop20BySessionIdOrderByCreatedAtDesc(Long sessionId);
}
