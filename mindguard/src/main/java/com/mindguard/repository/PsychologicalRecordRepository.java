package com.mindguard.repository;

import com.mindguard.model.PsychologicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface PsychologicalRecordRepository extends JpaRepository<PsychologicalRecord, Long> {
    List<PsychologicalRecord> findByUserIdOrderByRecordedAtDesc(Long userId);
}
