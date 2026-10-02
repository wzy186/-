package com.dss.controller;

import com.dss.dto.Result;
import com.dss.entity.Voucher;
import com.dss.service.IVoucherService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/voucher")
@RequiredArgsConstructor
public class VoucherController {

    private final IVoucherService voucherService;

    @GetMapping("/list/{shopId}")
    public Result queryVoucherOfShop(@PathVariable Long shopId) {
        return voucherService.queryVoucherOfShop(shopId);
    }

    @PostMapping
    public Result addVoucher(@RequestBody Voucher voucher) {
        return voucherService.addVoucher(voucher);
    }

    @GetMapping("/seckill/list/{shopId}")
    public Result querySeckillList(@PathVariable Long shopId) {
        return voucherService.querySeckillList(shopId);
    }

    /**
     * 手动预热某秒杀券库存到 Redis(测试用)
     */
    @PostMapping("/seckill/preheat/{voucherId}")
    public Result preheat(@PathVariable Long voucherId) {
        voucherService.preheatSeckillStock(voucherId);
        return Result.ok();
    }
}
