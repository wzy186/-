-- MindGuard 数据库初始化脚本

CREATE TABLE IF NOT EXISTS users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    real_name VARCHAR(100),
    email VARCHAR(200),
    student_id VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    final_risk_level VARCHAR(20),
    summary TEXT,
    archived BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    role VARCHAR(30) NOT NULL,
    content TEXT,
    tool_name VARCHAR(100),
    tool_args TEXT,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS psychological_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    session_id BIGINT NOT NULL,
    emotion_label VARCHAR(30),
    risk_level VARCHAR(20),
    conversation_summary TEXT,
    key_phrases TEXT,
    email_sent BOOLEAN DEFAULT FALSE,
    recorded_at DATETIME NOT NULL
);

-- 索引
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_user ON chat_messages(user_id);
CREATE INDEX idx_psychological_records_user ON psychological_records(user_id);