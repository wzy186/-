package com.dss.utils;

/**
 * Redis Key 常量
 */
public class RedisConstants {

    public static final String PREFIX = "dss:";

    // 登录验证码:login:code:{手机号} -> 6位验证码
    public static final String LOGIN_CODE_KEY = PREFIX + "login:code:";
    public static final long LOGIN_CODE_TTL = 120; // 秒

    // 登录 Token:login:token:{token} -> 用户Hash
    public static final String LOGIN_TOKEN_KEY = PREFIX + "login:token:";
    public static final long LOGIN_TOKEN_TTL = 1800; // 30 分钟

    // 商铺缓存:cache:shop:{id} -> Shop JSON
    public static final String CACHE_SHOP_KEY = PREFIX + "cache:shop:";
    public static final long CACHE_SHOP_TTL = 30; // 分钟
    // 空值缓存,防穿透
    public static final String CACHE_SHOP_NULL_KEY = PREFIX + "cache:shop:null:";
    public static final long CACHE_NULL_TTL = 2; // 分钟

    // 逻辑过期(防击穿):cache:shop:logic:{id} -> ShopDTO(含 expire 字段)
    public static final String CACHE_SHOP_LOGIC_KEY = PREFIX + "cache:shop:logic:";
    // 互斥锁:lock:shop:{id} -> 1
    public static final String LOCK_SHOP_KEY = PREFIX + "lock:shop:";
    public static final long LOCK_SHOP_TTL = 10; // 秒

    // 全局唯一 ID:id:seq:{业务前缀}:{日期} -> 自增
    public static final String ID_SEQ_KEY = PREFIX + "id:seq:";

    // 秒杀券库存:seckill:stock:{voucherId}
    public static final String SECKILL_STOCK_KEY = PREFIX + "seckill:stock:";
    // 秒杀券已购用户:seckill:order:{voucherId} -> Set(uid)
    public static final String SECKILL_ORDER_KEY = PREFIX + "seckill:order:";
    // 秒杀下单异步队列
    public static final String SECKILL_STREAM_KEY = PREFIX + "stream:seckill:order";
    public static final String SECKILL_STREAM_GROUP = "g1";

    // 红包金额 List:redpacket:{id}:amounts
    public static final String REDPACKET_AMOUNTS_KEY = PREFIX + "redpacket:";
    public static final String REDPACKET_AMOUNTS_SUFFIX = ":amounts";
    // 红包元数据 Hash:redpacket:{id}:meta
    public static final String REDPACKET_META_SUFFIX = ":meta";
    // 红包已领记录 Hash:redpacket:{id}:taken
    public static final String REDPACKET_TAKEN_SUFFIX = ":taken";
    // 红包序号
    public static final String REDPACKET_SEQ_KEY = PREFIX + "redpacket:seq";
    // 红包用户限流:redpacket:rate:{redpacketId}:{uid}
    public static final String REDPACKET_RATE_KEY = PREFIX + "redpacket:rate:";
    // 红包未领退款延迟队列
    public static final String REDPACKET_REFUND_QUEUE_KEY = PREFIX + "redpacket:refund:queue";

    // 点赞:blog:liked:{blogId} -> ZSet(uid, 时间戳)
    public static final String BLOG_LIKED_KEY = PREFIX + "blog:liked:";
    // 博客点赞数:blog:likes:{blogId}
    public static final String BLOG_LIKES_KEY = PREFIX + "blog:likes:";

    // 关注:follow:{uid} -> Set(targetUid)
    public static final String FOLLOW_KEY = PREFIX + "follow:";
    // Feed 流收件箱:feed:{uid} -> ZSet(msgId, 时间戳)
    public static final String FEED_KEY = PREFIX + "feed:";

    // 附近商铺 GEO:geo:shop
    public static final String SHOP_GEO_KEY = PREFIX + "geo:shop";

    // 签到:sign:{uid}:{yyyyMM} -> BitMap
    public static final String SIGN_KEY = PREFIX + "sign:";

    // UV 统计:uv:{bizKey} -> HyperLogLog
    public static final String UV_KEY = PREFIX + "uv:";

    private RedisConstants() {}
}
