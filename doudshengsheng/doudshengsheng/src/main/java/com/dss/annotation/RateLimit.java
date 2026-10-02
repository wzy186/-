package com.dss.annotation;

import java.lang.annotation.*;

/**
 * 接口限流注解:基于 Redisson 令牌桶,按 IP 维度限流。
 * <p>
 * 用法:在 Controller 方法上加 @RateLimit(rate=10, interval=1)
 * 含义:每个 IP 每秒最多 10 次请求,超出返回 429。
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface RateLimit {

    /** 令牌桶容量(最多放多少令牌) */
    int rate() default 10;

    /** 时间窗口(秒) */
    int interval() default 1;

    /** 限流维度 key,默认按 IP */
    String key() default "ip";
}
