package com.dss.controller;

import com.dss.annotation.AdminOnly;
import com.dss.dto.Result;
import com.dss.entity.Voucher;
import com.dss.service.IRedPacketService;
import com.dss.service.IVoucherService;
import com.dss.service.IVoucherOrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

/**
 * 商户后台管理接口。
 * 注意:演示用,未做管理员权限校验;生产应加管理员鉴权。
 */
@Tag(name = "商户后台", description = "秒杀券管理、红包雨数据、订单统计")
@RestController
@RequestMapping("/admin")
@RequiredArgsConstructor
public class AdminController {

    private final IVoucherService voucherService;
    private final IRedPacketService redPacketService;
    private final IVoucherOrderService voucherOrderService;
    private final com.dss.service.IShopService shopService;

    // ===== 商铺管理 =====

    @Operation(summary = "新增商铺(自动加布隆过滤器+GEO)")
    @AdminOnly
    @PostMapping("/shop")
    public Result addShop(@RequestBody com.dss.entity.Shop shop) {
        return shopService.save(shop);
    }

    // ===== 秒杀券管理 =====

    @Operation(summary = "创建优惠券(秒杀券自动预热库存)")
    @AdminOnly
    @PostMapping("/voucher")
    public Result addVoucher(@RequestBody Voucher voucher) {
        return voucherService.addVoucher(voucher);
    }

    @Operation(summary = "秒杀券库存(Redis 实时)")
    @AdminOnly
    @GetMapping("/voucher/stock/{id}")
    public Result voucherStock(@PathVariable Long id) {
        return voucherService.queryStock(id);
    }

    @Operation(summary = "重新预热秒杀券库存")
    @AdminOnly
    @PostMapping("/voucher/preheat/{id}")
    public Result preheat(@PathVariable Long id) {
        voucherService.preheatSeckillStock(id);
        return Result.ok();
    }

    // ===== 红包雨数据 =====

    @Operation(summary = "红包雨场次列表")
    @AdminOnly
    @GetMapping("/redpacket/list")
    public Result redPacketList() {
        return redPacketService.listRedPackets();
    }

    @Operation(summary = "红包雨场次实时详情")
    @AdminOnly
    @GetMapping("/redpacket/{id}")
    public Result redPacketDetail(@PathVariable Long id) {
        return redPacketService.redPacketDetail(id);
    }

    // ===== 订单统计 =====

    @Operation(summary = "秒杀订单统计")
    @AdminOnly
    @GetMapping("/order/stats")
    public Result orderStats() {
        return voucherOrderService.orderStats();
    }
}
