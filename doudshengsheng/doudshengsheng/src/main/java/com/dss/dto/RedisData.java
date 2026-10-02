package com.dss.dto;

import lombok.Data;

import java.time.LocalDateTime;

/**
 * 包装缓存数据 + 逻辑过期时间,用于防击穿的逻辑过期方案
 */
@Data
public class RedisData {
    private LocalDateTime expireTime;
    private Object data;
}
