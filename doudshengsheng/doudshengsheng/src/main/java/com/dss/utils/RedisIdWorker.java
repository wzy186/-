package com.dss.utils;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;

/**
 * 全局唯一 ID 生成器
 * <p>
 * 结构:符号位(1) + 时间戳(31) + 序列号(32)
 * 时间戳从 2024-01-01 起算,秒级;序列号用 Redis INCR 自增,按"业务前缀+天"分 key 避免溢出
 */
@Component
public class RedisIdWorker {

    private static final long BEGIN_TIMESTAMP = LocalDateTime.of(2024, 1, 1, 0, 0, 0)
            .toEpochSecond(ZoneOffset.UTC);

    private final StringRedisTemplate redis;

    public RedisIdWorker(StringRedisTemplate redis) {
        this.redis = redis;
    }

    /**
     * @param keyPrefix 业务前缀,如 voucher / order
     */
    public long nextId(String keyPrefix) {
        // 1. 时间戳
        LocalDateTime now = LocalDateTime.now();
        long nowSecond = now.toEpochSecond(ZoneOffset.UTC);
        long timestamp = nowSecond - BEGIN_TIMESTAMP;

        // 2. 序列号:用日期做 key 一部分,便于统计且避免单 key 过大
        String date = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        long count = redis.opsForValue().increment(RedisConstants.ID_SEQ_KEY + keyPrefix + ":" + date);

        // 3. 拼接:时间戳左移 32 位,低位放序列号
        return timestamp << 32 | count;
    }
}
