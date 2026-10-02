package com.dss.utils;

import cn.hutool.core.util.BooleanUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.dss.dto.RedisData;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * 缓存客户端:封装旁路缓存 + 三大问题(穿透/击穿/雪崩)
 */
@Slf4j
@Component
public class CacheClient {

    private final StringRedisTemplate redis;

    // 独立线程池,缓存重建用,避免阻塞 Tomcat 线程
    private static final ExecutorService CACHE_REBUILD_EXECUTOR = Executors.newFixedThreadPool(10);

    public CacheClient(StringRedisTemplate redis) {
        this.redis = redis;
    }

    // ===================== 基础方法 =====================

    /**
     * 写缓存,带 TTL
     */
    public void set(String key, Object value, Long ttl, TimeUnit unit) {
        redis.opsForValue().set(key, JSONUtil.toJsonStr(value), ttl, unit);
    }

    /**
     * 写缓存,带逻辑过期(不设 Redis TTL,值里带 expire 字段),用于防击穿
     */
    public void setWithLogicalExpire(String key, Object value, Long ttl, TimeUnit unit) {
        RedisData redisData = new RedisData();
        redisData.setData(value);
        redisData.setExpireTime(LocalDateTime.now().plusSeconds(unit.toSeconds(ttl)));
        redis.opsForValue().set(key, JSONUtil.toJsonStr(redisData));
    }

    /**
     * 普通旁路缓存查询。
     * 防雪崩:TTL 加随机数(调用方传随机 ttl)。
     * 防穿透:查不到的也缓存空值。
     */
    public <R, ID> R queryWithPassThrough(
            String keyPrefix, ID id, Class<R> type, Function<ID, R> dbFallback, Long ttl, TimeUnit unit) {
        String key = keyPrefix + id;
        String json = redis.opsForValue().get(key);

        // 1. 缓存命中
        if (StrUtil.isNotBlank(json)) {
            return JSONUtil.toBean(json, type);
        }
        // 2. 命中的是空值(防穿透)
        if (json != null) { // 空字符串 ""
            return null;
        }

        // 3. 缓存未命中,查 DB
        R r = dbFallback.apply(id);
        if (r == null) {
            // 3.1 空值缓存,短 TTL 防穿透
            redis.opsForValue().set(key, "", RedisConstants.CACHE_NULL_TTL, TimeUnit.MINUTES);
            return null;
        }
        // 3.2 写缓存
        this.set(key, r, ttl, unit);
        return r;
    }

    // ===================== 防击穿:互斥锁方案 =====================

    /**
     * 互斥锁:缓存未命中时,只放一个线程重建缓存,其他线程等待重试。
     * 优点:一致性高;缺点:吞吐降。
     */
    public <R, ID> R queryWithMutex(
            String keyPrefix, ID id, Class<R> type, Function<ID, R> dbFallback, Long ttl, TimeUnit unit) {
        String key = keyPrefix + id;
        String json = redis.opsForValue().get(key);

        // 1. 命中
        if (StrUtil.isNotBlank(json)) {
            return JSONUtil.toBean(json, type);
        }
        if (json != null) {
            return null; // 空值
        }

        // 2. 未命中,尝试拿互斥锁
        String lockKey = RedisConstants.LOCK_SHOP_KEY + id;
        R r = null;
        try {
            boolean lock = tryLock(lockKey);
            if (!lock) {
                // 2.1 拿不到锁,休眠重试
                Thread.sleep(50);
                return queryWithMutex(keyPrefix, id, type, dbFallback, ttl, unit);
            }
            // 2.2 双重检查:可能前一个线程已重建
            json = redis.opsForValue().get(key);
            if (StrUtil.isNotBlank(json)) {
                return JSONUtil.toBean(json, type);
            }
            // 3. 查 DB 重建
            r = dbFallback.apply(id);
            if (r == null) {
                redis.opsForValue().set(key, "", RedisConstants.CACHE_NULL_TTL, TimeUnit.MINUTES);
                return null;
            }
            this.set(key, r, ttl, unit);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            unLock(lockKey);
        }
        return r;
    }

    // ===================== 防击穿:逻辑过期方案 =====================

    /**
     * 逻辑过期:缓存永不过期(Redis 层面),值里带 expireTime。
     * 读到过期时,放一个线程异步重建(拿锁),当前线程返回旧数据。
     * 优点:吞吐高不等待;缺点:短时不一致 + 需预热。
     */
    public <R, ID> R queryWithLogicalExpire(
            String keyPrefix, ID id, Class<R> type, Function<ID, R> dbFallback, Long ttl, TimeUnit unit) {
        String key = keyPrefix + id;
        String json = redis.opsForValue().get(key);

        // 1. 未命中(逻辑过期方案需提前预热,没数据直接返回 null)
        if (StrUtil.isBlank(json)) {
            return null;
        }
        // 2. 命中,反序列化
        RedisData redisData = JSONUtil.toBean(json, RedisData.class);
        R r = JSONUtil.toBean((JSONObject) redisData.getData(), type);
        LocalDateTime expireTime = redisData.getExpireTime();

        // 3. 未过期,直接返回
        if (expireTime.isAfter(LocalDateTime.now())) {
            return r;
        }

        // 4. 已过期,尝试拿锁异步重建
        String lockKey = RedisConstants.LOCK_SHOP_KEY + id;
        if (tryLock(lockKey)) {
            // 拿到锁,开异步线程重建(双重检查后再重建)
            CACHE_REBUILD_EXECUTOR.submit(() -> {
                try {
                    // 双重检查
                    String again = redis.opsForValue().get(key);
                    if (StrUtil.isNotBlank(again)) {
                        RedisData againData = JSONUtil.toBean(again, RedisData.class);
                        if (againData.getExpireTime().isAfter(LocalDateTime.now())) {
                            return;
                        }
                    }
                    R newR = dbFallback.apply(id);
                    this.setWithLogicalExpire(key, newR, ttl, unit);
                    log.debug("缓存重建完成 key={}", key);
                } finally {
                    unLock(lockKey);
                }
            });
        }
        // 5. 无论是否拿到锁,都返回旧数据
        return r;
    }

    // ===================== 锁工具 =====================

    private boolean tryLock(String key) {
        Boolean flag = redis.opsForValue().setIfAbsent(key, "1", RedisConstants.LOCK_SHOP_TTL, TimeUnit.SECONDS);
        return BooleanUtil.isTrue(flag);
    }

    private void unLock(String key) {
        redis.delete(key);
    }
}
