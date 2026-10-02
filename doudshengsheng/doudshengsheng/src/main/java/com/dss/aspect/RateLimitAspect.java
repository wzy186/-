package com.dss.aspect;

import com.dss.annotation.RateLimit;
import com.dss.exception.BizException;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.redisson.api.RRateLimiter;
import org.redisson.api.RateIntervalUnit;
import org.redisson.api.RateType;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

/**
 * 限流切面:基于 Redisson 分布式令牌桶。
 * <p>
 * 令牌桶特点:允许突发(桶里有存量令牌时快速放行),平滑限流。
 * 分布式:多实例共享同一个 Redis 限流器,全局限流。
 */
@Slf4j
@Aspect
@Component
@RequiredArgsConstructor
public class RateLimitAspect {

    private final RedissonClient redisson;

    @Around("@annotation(rateLimit)")
    public Object around(ProceedingJoinPoint pjp, RateLimit rateLimit) throws Throwable {
        String clientKey = resolveKey(rateLimit.key());
        // 限流器 key:按 IP + 方法签名隔离
        String methodName = pjp.getSignature().toShortString();
        String limiterKey = "dss:ratelimit:" + methodName + ":" + clientKey;

        RRateLimiter limiter = redisson.getRateLimiter(limiterKey);
        // 初始化令牌桶(trySetRate 不会覆盖已有配置)
        limiter.trySetRate(RateType.OVERALL, rateLimit.rate(), rateLimit.interval(), RateIntervalUnit.SECONDS);

        // 非阻塞尝试获取 1 个令牌
        if (!limiter.tryAcquire(1)) {
            log.warn("限流触发 key={} ip={}", limiterKey, clientKey);
            throw new BizException(429, "请求过于频繁,请稍后再试");
        }
        return pjp.proceed();
    }

    /**
     * 解析限流维度 key,目前支持 IP。
     * 取真实 IP(穿透代理):优先 X-Forwarded-For / X-Real-IP。
     */
    private String resolveKey(String dimension) {
        ServletRequestAttributes attrs = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attrs == null) {
            return "unknown";
        }
        HttpServletRequest req = attrs.getRequest();
        String ip = req.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = req.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = req.getRemoteAddr();
        }
        // X-Forwarded-For 可能是 "client, proxy1, proxy2",取第一个
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip == null ? "unknown" : ip;
    }
}
