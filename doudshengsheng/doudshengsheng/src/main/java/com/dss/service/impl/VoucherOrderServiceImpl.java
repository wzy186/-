package com.dss.service.impl;

import cn.hutool.core.lang.UUID;
import com.dss.dto.Result;
import com.dss.entity.SeckillVoucher;
import com.dss.entity.VoucherOrder;
import com.dss.mapper.SeckillVoucherMapper;
import com.dss.mapper.VoucherOrderMapper;
import com.dss.service.IVoucherOrderService;
import com.dss.utils.RedisConstants;
import com.dss.utils.RedisIdWorker;
import com.dss.utils.UserHolder;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.core.io.ClassPathResource;
import org.springframework.data.redis.connection.stream.*;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * 秒杀下单。
 * <p>
 * 流程:
 * 1. Lua 脚本原子校验(时间 + 库存 + 一人一单)并扣库存
 * 2. 校验通过 → 生成订单 ID → 发到 Redis Stream 异步落库
 * 3. 消费者:分布式锁兜底一人一单(双保险),事务内先乐观锁扣 DB 库存(stock > 0 CAS 防超卖)
 *    再写订单,任一步失败整体回滚,最后 ACK
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class VoucherOrderServiceImpl implements IVoucherOrderService {

    private final SeckillVoucherMapper seckillVoucherMapper;
    private final VoucherOrderMapper voucherOrderMapper;
    private final RedisIdWorker idWorker;
    private final StringRedisTemplate redis;
    private final RedissonClient redisson;
    private final TransactionTemplate txTemplate;

    private DefaultRedisScript<Long> seckillScript;
    private final ExecutorService orderExecutor = Executors.newSingleThreadExecutor();

    @PostConstruct
    public void init() {
        seckillScript = new DefaultRedisScript<>();
        seckillScript.setLocation(new ClassPathResource("lua/seckill.lua"));
        seckillScript.setResultType(Long.class);
        // 启动消费线程
        orderExecutor.submit(this::consumeStream);
    }

    @Override
    public Result seckillVoucher(Long voucherId) {
        Long userId = UserHolder.getUserId();
        if (userId == null) {
            return Result.fail(401, "未登录");
        }
        SeckillVoucher sv = seckillVoucherMapper.selectById(voucherId);
        if (sv == null) {
            return Result.fail("秒杀券不存在");
        }
        long now = System.currentTimeMillis();
        Long r = redis.execute(
                seckillScript,
                List.of(
                        RedisConstants.SECKILL_STOCK_KEY + voucherId,
                        RedisConstants.SECKILL_ORDER_KEY + voucherId),
                userId.toString(),
                String.valueOf(now),
                String.valueOf(toMillis(sv.getBeginTime())),
                String.valueOf(toMillis(sv.getEndTime()))
        );

        int result = r == null ? -1 : r.intValue();
        if (result != 0) {
            return Result.fail(codeMsg(result));
        }
        // 校验通过,生成订单,发到 Stream
        long orderId = idWorker.nextId("order");
        redis.opsForStream().add(RedisConstants.SECKILL_STREAM_KEY,
                Map.of(
                        "orderId", String.valueOf(orderId),
                        "userId", String.valueOf(userId),
                        "voucherId", String.valueOf(voucherId)));
        return Result.ok(orderId);
    }

    /**
     * 消费 Stream,异步落库。
     */
    private void consumeStream() {
        String streamKey = RedisConstants.SECKILL_STREAM_KEY;
        String group = RedisConstants.SECKILL_STREAM_GROUP;
        String consumer = "c1-" + UUID.randomUUID().toString(true);
        try {
            redis.opsForStream().createGroup(streamKey, group);
        } catch (Exception ignore) {
            // 组已存在
        }
        while (!Thread.currentThread().isInterrupted()) {
            try {
                List<MapRecord<String, Object, Object>> list = redis.opsForStream().read(
                        Consumer.from(group, consumer),
                        StreamReadOptions.empty().count(1).block(Duration.ofSeconds(2)),
                        StreamOffset.create(streamKey, ReadOffset.lastConsumed()));
                if (list == null || list.isEmpty()) {
                    continue;
                }
                for (MapRecord<String, Object, Object> record : list) {
                    handleRecord(record, streamKey, group);
                }
            } catch (Exception e) {
                String msg = e.getMessage() == null ? "" : e.getMessage();
                log.warn("Stream 读取异常: {}", msg);
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

    /**
     * 处理一条消息:分布式锁 + 显式事务落库 + ACK。
     * 用 TransactionTemplate 而非 @Transactional,因为消费线程自调用不会走代理。
     */
    private void handleRecord(MapRecord<String, Object, Object> record, String streamKey, String group) {
        Map<Object, Object> val = record.getValue();
        Long orderId = Long.valueOf(val.get("orderId").toString());
        Long userId = Long.valueOf(val.get("userId").toString());
        Long voucherId = Long.valueOf(val.get("voucherId").toString());

        RLock lock = redisson.getLock("dss:lock:order:" + userId);
        boolean locked = false;
        try {
            locked = lock.tryLock(3, 10, TimeUnit.SECONDS);
            if (!locked) {
                // Lua 已挡住,这里兜底;直接 ACK 丢弃
                redis.opsForStream().acknowledge(streamKey, group, record.getId());
                return;
            }
            Boolean ok = txTemplate.execute(status -> {
                // 乐观锁扣 DB 库存:stock > 0 的 CAS 条件兜底超卖(最后一道防线)。
                // 失败说明 Redis 与 DB 库存已不一致(Redis 侧被 Lua 扣过),不建单,避免 DB 超发。
                int rows = seckillVoucherMapper.deductStock(voucherId);
                if (rows == 0) {
                    status.setRollbackOnly();
                    return false;
                }
                VoucherOrder order = new VoucherOrder();
                order.setId(orderId);
                order.setUserId(userId);
                order.setVoucherId(voucherId);
                order.setPayType(1);
                order.setStatus(2);
                order.setCreateTime(LocalDateTime.now());
                voucherOrderMapper.insert(order);
                return true;
            });
            redis.opsForStream().acknowledge(streamKey, group, record.getId());
            if (Boolean.FALSE.equals(ok)) {
                // 乐观锁拦截:DB 库存不足。消息已 ACK,不再重试(重试也会失败)。
                log.error("乐观锁拦截,DB库存不足,未建单 orderId={} userId={} voucherId={}", orderId, userId, voucherId);
            } else {
                log.debug("落库成功 orderId={}", orderId);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (locked && lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }

    @Override
    public Result orderStats() {
        // 订单总数
        Long total = voucherOrderMapper.selectCount(null);
        // 按券分组统计
        List<java.util.Map<String, Object>> byVoucher = voucherOrderMapper.selectMaps(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<VoucherOrder>()
                        .select("voucher_id", "count(*) as cnt")
                        .groupBy("voucher_id"));
        java.util.Map<String, Object> data = new java.util.HashMap<>();
        data.put("total", total);
        data.put("byVoucher", byVoucher);
        return Result.ok(data);
    }

    @Override
    public Result myOrders() {
        Long userId = UserHolder.getUserId();
        if (userId == null) return Result.fail(401, "未登录");
        List<VoucherOrder> orders = voucherOrderMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<VoucherOrder>()
                        .eq(VoucherOrder::getUserId, userId)
                        .orderByDesc(VoucherOrder::getCreateTime));
        return Result.ok(orders);
    }

    private String codeMsg(int code) {
        return switch (code) {
            case 1 -> "活动未开始";
            case 2 -> "活动已结束";
            case 3 -> "库存不足";
            case 4 -> "不可重复下单";
            default -> "下单失败";
        };
    }

    private long toMillis(LocalDateTime t) {
        return t == null ? 0L : t.atZone(ZoneOffset.UTC).toInstant().toEpochMilli();
    }
}
