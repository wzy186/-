package com.mindguard.repository;

import com.mindguard.model.ChatSession;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;

public interface ChatSessionRepository extends JpaRepository<ChatSession, Long> {
    Optional<ChatSession> findTopByUserIdAndEndTimeIsNullOrderByStartTimeDesc(Long userId);
    List<ChatSession> findByUserIdOrderByStartTimeDesc(Long userId);
}
