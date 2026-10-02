package com.dss.service.impl;

import cn.hutool.core.lang.UUID;
import com.dss.dto.Result;
import com.dss.entity.RedPacket;
import com.dss.entity.RedPacketRecord;
import com.dss.mapper.RedPacketMapper;
import com.dss.mapper.RedPacketRecordMapper;
import com.dss.service.IRedPacketService;
import com.dss.utils.RedPacketSplitter;
import com.dss.utils.RedisConstants;
import com.dss.utils.RedisIdWorker;
import com.dss.utils.UserHolder;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.data.redis.connection.stream.*;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.*;

/**
 * 红包雨服务。
 * <p>
 * 技术点:
 * 1. 二倍均值法拆金额
 * 2. 预分配金额到 Redis List,抢时 RPOP,O(1)
 * 3. Lua 原子抢红包(幂等 + 取金额 + 记录 + 减余量)
 * 4. 滑动窗口 ZSet+Lua 限流防刷
 * 5. Stream 异步落库削峰
 * 6. 延迟队列(定时扫描)未领退款
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RedPacketServiceImpl implements IRedPacketService {

    private final RedPacketMapper redPacketMapper;
    private final RedPacketRecordMapper recordMapper;
    private final RedisIdWorker idWorker;
    private final StringRedisTemplate redis;
    private final TransactionTemplate txTemplate;

    @Value("${dss.redpacket.refund-delay-seconds:120}")
    private long refundDelaySeconds;
    @Value("${dss.redpacket.rate-limit-window-seconds:10}")
    private long rateLimitWindowSeconds;
    @Value("${dss.redpacket.rate-limit-max:3}")
    private long rateLimitMax;

    private DefaultRedisScript<String> grabScript;
    private DefaultRedisScript<Long> rateLimitScript;

    private final ScheduledExecutorService refundScheduler = Executors.newSingleThreadScheduledExecutor();
    private final ExecutorService recordExecutor = Executors.newSingleThreadExecutor();

    @PostConstruct
    public void init() {
        // 抢红包 Lua
        grabScript = new DefaultRedisScript<>();
        grabScript.setLocation(new ClassPathResource("lua/redpacket_grab.lua"));
        grabScript.setResultType(String.class);

        // 限流 Lua
        rateLimitScript = new DefaultRedisScript<>();
        rateLimitScript.setLocation(new ClassPathResource("lua/sliding_window.lua"));
        rateLimitScript.setResultType(Long.class);

        // 启动红包领取记录异步落库消费者
        recordExecutor.submit(this::consumeRecordStream);
    }

    // ===================== 创建红包雨 =====================

    @Override
    public Result createRedPacket(String title, int totalYuan, int count) {
        if (totalYuan <= 0 || count <= 0 || totalYuan < count) {
            return Result.fail("参数非法:金额和个数必须为正,且金额(分)不能小于个数");
        }
        int totalCents = totalYuan * 100;
        // 1. 二倍均值法拆金额
        List<Integer> amounts = RedPacketSplitter.split(totalCents, count);

        // 2. 生成红包 id
        long id = idWorker.nextId("redpacket");

        // 3. 预分配金额到 Redis List
        String amountsKey = RedisConstants.REDPACKET_AMOUNTS_KEY + id + RedisConstants.REDPACKET_AMOUNTS_SUFFIX;
        redis.opsForList().rightPushAll(amountsKey,
                amounts.stream().map(String::valueOf).toList());

        // 4. 元数据 Hash
        String metaKey = RedisConstants.REDPACKET_AMOUNTS_KEY + id + RedisConstants.REDPACKET_META_SUFFIX;
        Map<String, String> meta = new HashMap<>();
        meta.put("total", String.valueOf(totalCents));
        meta.put("count", String.valueOf(count));
        meta.put("remain", String.valueOf(count));
        meta.put("got", "0");
        meta.put("status", "1");
        meta.put("expireAt", String.valueOf(System.currentTimeMillis() + refundDelaySeconds * 1000));
        redis.opsForHash().putAll(metaKey, meta);

        // 5. 设置 TTL:到期后再多留 60s 给退款任务读取,之后自动清理
        long ttl = refundDelaySeconds + 60;
        redis.expire(amountsKey, ttl, TimeUnit.SECONDS);
        redis.expire(metaKey, ttl, TimeUnit.SECONDS);
        redis.expire(RedisConstants.REDPACKET_AMOUNTS_KEY + id + RedisConstants.REDPACKET_TAKEN_SUFFIX,
                ttl, TimeUnit.SECONDS);

        // 6. 入延迟队列:refundDelaySeconds 后检查退款
        refundScheduler.schedule(() -> refundUnclaimed(id), refundDelaySeconds, TimeUnit.SECONDS);

        // 7. 落库
        RedPacket rp = new RedPacket();
        rp.setId(id);
        rp.setTitle(title);
        rp.setTotalAmount((long) totalCents);
        rp.setCount(count);
        rp.setRemainCount(count);
        rp.setGotCount(0);
        rp.setStatus(1);
        rp.setCreateTime(LocalDateTime.now());
        redPacketMapper.insert(rp);

        log.info("红包雨创建 id={} title={} total={}cents count={}", id, title, totalCents, count);
        return Result.ok(id);
    }

    // ===================== 抢红包 =====================

    @Override
    public Result grab(Long redPacketId) {
        Long userId = UserHolder.getUserId();
        if (userId == null) {
            return Result.fail(401, "未登录");
        }
        // 1. 限流防刷(滑动窗口)
        String rateKey = RedisConstants.REDPACKET_RATE_KEY + redPacketId + ":" + userId;
        String member = System.currentTimeMillis() + ":" + UUID.randomUUID().toString(true);
        Long allowed = redis.execute(rateLimitScript,
                List.of(rateKey),
                String.valueOf(System.currentTimeMillis()),
                String.valueOf(rateLimitWindowSeconds * 1000),
                String.valueOf(rateLimitMax),
                member);
        if (allowed == null || allowed == 0) {
            return Result.fail(429, "操作太频繁,请稍后再试");
        }

        // 2. Lua 原子抢红包
        String amountsKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_AMOUNTS_SUFFIX;
        String takenKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_TAKEN_SUFFIX;
        String metaKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_META_SUFFIX;
        String result = redis.execute(grabScript,
                List.of(amountsKey, takenKey, metaKey),
                userId.toString(),
                String.valueOf(System.currentTimeMillis() / 1000));

        if (result == null) {
            return Result.fail("红包异常");
        }
        if ("-1".equals(result)) {
            return Result.fail("您已领过该红包");
        }
        if ("0".equals(result)) {
            return Result.fail("红包已抢完");
        }

        long amount = Long.parseLong(result);
        // 3. 异步落库:发到 Stream
        redis.opsForStream().add(RedisConstants.PREFIX + "stream:redpacket:record",
                Map.of(
                        "redPacketId", String.valueOf(redPacketId),
                        "userId", String.valueOf(userId),
                        "amount", String.valueOf(amount)));
        return Result.ok(amount);
    }

    // ===================== 排行榜 =====================

    @Override
    public Result rank(Long redPacketId) {
        String takenKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_TAKEN_SUFFIX;
        Map<Object, Object> taken = redis.opsForHash().entries(takenKey);
        // 过滤掉 :time 的字段,组装 uid->amount
        List<Map<String, Object>> rank = new ArrayList<>();
        for (Map.Entry<Object, Object> e : taken.entrySet()) {
            String k = e.getKey().toString();
            if (k.endsWith(":time")) continue;
            Map<String, Object> item = new HashMap<>();
            item.put("userId", k);
            item.put("amount", e.getValue());
            rank.add(item);
        }
        // 按金额降序
        rank.sort((a, b) -> Long.compare(
                Long.parseLong(b.get("amount").toString()),
                Long.parseLong(a.get("amount").toString())));
        return Result.ok(rank);
    }

    // ===================== 后台:场次列表与详情 =====================

    @Override
    public Result listRedPackets() {
        // 按创建时间倒序查最近 50 场
        List<RedPacket> list = redPacketMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<RedPacket>()
                        .orderByDesc(RedPacket::getCreateTime)
                        .last("limit 50"));
        return Result.ok(list);
    }

    @Override
    public Result redPacketDetail(Long redPacketId) {
        Map<String, Object> resp = new HashMap<>();
        // DB 元数据
        RedPacket rp = redPacketMapper.selectById(redPacketId);
        resp.put("info", rp);
        // Redis 实时进度
        String metaKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_META_SUFFIX;
        Map<Object, Object> meta = redis.opsForHash().entries(metaKey);
        resp.put("meta", meta);
        // 已领记录数
        String takenKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_TAKEN_SUFFIX;
        Long takenCount = redis.opsForHash().size(takenKey);
        // taken 里每人有 amount + time 两个 field,实际领取人数减半
        resp.put("takenCount", takenCount == null ? 0 : takenCount / 2);
        return Result.ok(resp);
    }

    // ===================== 异步落库 =====================

    private void consumeRecordStream() {
        String streamKey = RedisConstants.PREFIX + "stream:redpacket:record";
        String group = "g1";
        String consumer = "c1-" + UUID.randomUUID().toString(true);
        try {
            redis.opsForStream().createGroup(streamKey, group);
        } catch (Exception ignore) {}
        while (!Thread.currentThread().isInterrupted()) {
            try {
                List<MapRecord<String, Object, Object>> list = redis.opsForStream().read(
                        Consumer.from(group, consumer),
                        StreamReadOptions.empty().count(50).block(Duration.ofSeconds(2)),
                        StreamOffset.create(streamKey, ReadOffset.lastConsumed()));
                if (list == null || list.isEmpty()) continue;
                List<RedPacketRecord> batch = new ArrayList<>();
                List<RecordId> ids = new ArrayList<>();
                for (MapRecord<String, Object, Object> rec : list) {
                    Map<Object, Object> v = rec.getValue();
                    RedPacketRecord r = new RedPacketRecord();
                    r.setRedPacketId(Long.valueOf(v.get("redPacketId").toString()));
                    r.setUserId(Long.valueOf(v.get("userId").toString()));
                    r.setAmount(Long.valueOf(v.get("amount").toString()));
                    r.setGrabTime(LocalDateTime.now());
                    batch.add(r);
                    ids.add(rec.getId());
                }
                // 批量落库
                txTemplate.execute(status -> {
                    for (RedPacketRecord r : batch) {
                        recordMapper.insert(r);
                    }
                    // 更新红包 gotCount(可优化为批量)
                    return null;
                });
                for (RecordId id : ids) {
                    redis.opsForStream().acknowledge(streamKey, group, id);
                }
                log.debug("红包记录批量落库 {} 条", batch.size());
            } catch (Exception e) {
                log.warn("红包记录消费异常: {}", e.getMessage());
                // 退避,避免异常时死循环打爆日志/拖垮服务
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    // ===================== 未领退款 =====================

    /**
     * 红包雨到期后,把未领的金额"退回"(演示:标记状态 + 记录剩余)。
     * 生产场景:退回发放方账户,这里只做状态变更和日志。
     */
    private void refundUnclaimed(long redPacketId) {
        try {
            String metaKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_META_SUFFIX;
            String amountsKey = RedisConstants.REDPACKET_AMOUNTS_KEY + redPacketId + RedisConstants.REDPACKET_AMOUNTS_SUFFIX;
            // 取剩余个数和已领
            Object remainObj = redis.opsForHash().get(metaKey, "remain");
            int remain = remainObj == null ? 0 : Integer.parseInt(remainObj.toString());
            if (remain > 0) {
                // 统计剩余金额
                List<String> leftAmounts = redis.opsForList().range(amountsKey, 0, -1);
                long refundAmount = 0;
                if (leftAmounts != null) {
                    for (String a : leftAmounts) refundAmount += Long.parseLong(a);
                }
                log.info("红包 {} 未领退款:剩余 {} 个,金额 {} 分", redPacketId, remain, refundAmount);
                // 更新 DB 状态
                RedPacket rp = redPacketMapper.selectById(redPacketId);
                if (rp != null) {
                    rp.setStatus(3);
                    rp.setEndTime(LocalDateTime.now());
                    rp.setRemainCount(remain);
                    redPacketMapper.updateById(rp);
                }
                redis.opsForHash().put(metaKey, "status", "3");
            } else {
                // 全部领完,标记抢完
                RedPacket rp = redPacketMapper.selectById(redPacketId);
                if (rp != null) {
                    rp.setStatus(2);
                    rp.setEndTime(LocalDateTime.now());
                    redPacketMapper.updateById(rp);
                }
                log.info("红包 {} 全部领完", redPacketId);
            }
        } catch (Exception e) {
            log.error("红包 {} 退款处理异常", redPacketId, e);
        }
    }
}
