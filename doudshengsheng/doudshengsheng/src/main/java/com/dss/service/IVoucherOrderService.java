package com.dss.service;

import com.dss.dto.Result;

public interface IVoucherOrderService {
    /** 秒杀下单:Lua 原子校验 + 分布式锁 + Stream 异步落库 */
    Result seckillVoucher(Long voucherId);
    /** 后台:订单统计 */
    Result orderStats();
    /** 我的订单 */
    Result myOrders();
}
