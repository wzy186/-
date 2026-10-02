package com.dss.service.impl;

import com.dss.dto.Result;
import com.dss.service.IStatsService;
import com.dss.utils.RedisConstants;
import com.dss.utils.UserHolder;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * 统计:签到(BitMap) + UV(HyperLogLog)。
 */
@Service
@RequiredArgsConstructor
public class StatsServiceImpl implements IStatsService {

    private final StringRedisTemplate redis;

    /**
     * 签到:BitMap,key=sign:{uid}:{yyyyMM},offset=日(1-31)。
     * 一个用户一个月只占 4 字节,极度省内存。
     */
    @Override
    public Result sign() {
        Long userId = UserHolder.getUserId();
        LocalDateTime now = LocalDateTime.now();
        String keySuffix = now.format(DateTimeFormatter.ofPattern("yyyyMM"));
        String key = RedisConstants.SIGN_KEY + userId + ":" + keySuffix;
        int dayOfMonth = now.getDayOfMonth();
        // SETBIT key offset 1
        redis.opsForValue().setBit(key, dayOfMonth - 1, true);
        return Result.ok();
    }

    /**
     * 连续签到天数:从今天往前数,遇到 0 停。
     * 用 BITFIELD 取最后 N 位,逐位判断。
     */
    @Override
    public Result signCount() {
        Long userId = UserHolder.getUserId();
        LocalDateTime now = LocalDateTime.now();
        String keySuffix = now.format(DateTimeFormatter.ofPattern("yyyyMM"));
        String key = RedisConstants.SIGN_KEY + userId + ":" + keySuffix;
        int dayOfMonth = now.getDayOfMonth();
        // BITFIELD key GET u{dayOfMonth} 0  —— 取从第 0 位起 dayOfMonth 位无符号
        // 这里用 BitFieldSubCommands
        org.springframework.data.redis.core.StringRedisTemplate t = redis;
        org.springframework.data.redis.connection.BitFieldSubCommands cmds =
                org.springframework.data.redis.connection.BitFieldSubCommands.create()
                        .get(org.springframework.data.redis.connection.BitFieldSubCommands.BitFieldType.unsigned(dayOfMonth))
                        .valueAt(0);
        java.util.List<Long> res = t.opsForValue().bitField(key, cmds);
        if (res == null || res.isEmpty() || res.get(0) == null || res.get(0) == 0) {
            return Result.ok(0);
        }
        long num = res.get(0);
        int count = 0;
        while ((num & 1) == 1) { // 最低位是今天
            count++;
            num >>>= 1;
        }
        return Result.ok(count);
    }

    /**
     * 本月签到记录:返回 List<Boolean>,下标对应日期(1-based)。
     * 用 GETBIT 逐位查,避免 BITFIELD 位顺序的坑(BITFIELD 的 bit0 是 MSB,容易搞反)。
     */
    @Override
    public Result signRecords() {
        Long userId = UserHolder.getUserId();
        LocalDateTime now = LocalDateTime.now();
        String keySuffix = now.format(DateTimeFormatter.ofPattern("yyyyMM"));
        String key = RedisConstants.SIGN_KEY + userId + ":" + keySuffix;
        int dayOfMonth = now.getDayOfMonth();

        java.util.List<Boolean> records = new java.util.ArrayList<>();
        records.add(false); // 下标0占位,让下标=日期
        for (int d = 1; d <= dayOfMonth; d++) {
            // BitMap offset = 日期 - 1
            Boolean signed = redis.opsForValue().getBit(key, d - 1);
            records.add(Boolean.TRUE.equals(signed));
        }
        return Result.ok(records);
    }

    /**
     * UV 统计:HyperLogLog,固定 12KB 内存,去重计数误差 0.81%。
     */
    @Override
    public Result uv(String bizKey, Long userId) {
        Long uid = userId != null ? userId : UserHolder.getUserId();
        redis.opsForHyperLogLog().add(RedisConstants.UV_KEY + bizKey, uid == null ? "anon" : uid.toString());
        return Result.ok();
    }

    @Override
    public Result uvCount(String bizKey) {
        long count = redis.opsForHyperLogLog().size(RedisConstants.UV_KEY + bizKey);
        return Result.ok(count);
    }
}
