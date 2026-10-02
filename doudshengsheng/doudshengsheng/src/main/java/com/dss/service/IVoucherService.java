package com.dss.service;

import com.dss.dto.Result;
import com.dss.entity.Voucher;

import java.util.List;

public interface IVoucherService {
    Result queryVoucherOfShop(Long shopId);
    Result addVoucher(Voucher voucher);
    Result querySeckillList(Long shopId);
    /** 把秒杀券库存预加载到 Redis */
    void preheatSeckillStock(Long voucherId);
    /** 查 Redis 里的秒杀券实时库存 */
    Result queryStock(Long voucherId);
}
