package com.dss.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.dto.Result;
import com.dss.entity.SeckillVoucher;
import com.dss.entity.Voucher;
import com.dss.mapper.SeckillVoucherMapper;
import com.dss.mapper.VoucherMapper;
import com.dss.service.IVoucherService;
import com.dss.utils.RedisConstants;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class VoucherServiceImpl implements IVoucherService {

    private final VoucherMapper voucherMapper;
    private final SeckillVoucherMapper seckillVoucherMapper;
    private final StringRedisTemplate redis;

    @Override
    public Result queryVoucherOfShop(Long shopId) {
        List<Voucher> list = voucherMapper.selectList(
                new LambdaQueryWrapper<Voucher>().eq(Voucher::getShopId, shopId));
        return Result.ok(list);
    }

    @Override
    @Transactional
    public Result addVoucher(Voucher voucher) {
        voucherMapper.insert(voucher);
        if (voucher.getType() != null && voucher.getType() == 2) {
            // 秒杀券:写附加表(库存演示固定 100)+ 预热到 Redis
            SeckillVoucher sv = new SeckillVoucher();
            sv.setVoucherId(voucher.getId());
            sv.setStock(100); // 演示用,真实场景从入参取
            seckillVoucherMapper.insert(sv);
            preheatSeckillStock(voucher.getId());
        }
        return Result.ok(voucher.getId());
    }

    @Override
    public Result querySeckillList(Long shopId) {
        List<Voucher> list = voucherMapper.selectList(
                new LambdaQueryWrapper<Voucher>()
                        .eq(Voucher::getShopId, shopId)
                        .eq(Voucher::getType, 2));
        return Result.ok(list);
    }

    @Override
    public void preheatSeckillStock(Long voucherId) {
        SeckillVoucher sv = seckillVoucherMapper.selectById(voucherId);
        if (sv == null) {
            return;
        }
        redis.opsForValue().set(RedisConstants.SECKILL_STOCK_KEY + voucherId, String.valueOf(sv.getStock()));
        log.info("秒杀券 {} 库存预热到 Redis: {}", voucherId, sv.getStock());
    }

    @Override
    public Result queryStock(Long voucherId) {
        String stock = redis.opsForValue().get(RedisConstants.SECKILL_STOCK_KEY + voucherId);
        SeckillVoucher sv = seckillVoucherMapper.selectById(voucherId);
        java.util.Map<String, Object> data = new java.util.HashMap<>();
        data.put("redisStock", stock == null ? 0 : Integer.parseInt(stock));
        data.put("dbStock", sv == null ? 0 : sv.getStock());
        return Result.ok(data);
    }
}
