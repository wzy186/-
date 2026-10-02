package com.dss.controller;

import com.dss.annotation.RateLimit;
import com.dss.dto.Result;
import com.dss.service.IVoucherOrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "优惠券订单", description = "秒杀下单")
@RestController
@RequestMapping("/voucher/order")
@RequiredArgsConstructor
public class VoucherOrderController {

    private final IVoucherOrderService voucherOrderService;

    /**
     * 秒杀下单
     */
    @Operation(summary = "秒杀下单", description = "Lua 原子校验扣库存 + Redisson 分布式锁 + Stream 异步落库")
    @RateLimit(rate = 5, interval = 1) // 每个 IP 每秒最多 5 次
    @PostMapping("/seckill/{id}")
    public Result seckillVoucher(@Parameter(description = "秒杀券 ID") @PathVariable("id") Long voucherId) {
        return voucherOrderService.seckillVoucher(voucherId);
    }

    @Operation(summary = "我的秒杀订单")
    @GetMapping("/my")
    public Result myOrders() {
        return voucherOrderService.myOrders();
    }
}
