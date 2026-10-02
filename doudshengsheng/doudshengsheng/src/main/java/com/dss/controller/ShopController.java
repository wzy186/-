package com.dss.controller;

import com.dss.dto.Result;
import com.dss.entity.Shop;
import com.dss.service.IShopService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Tag(name = "商铺", description = "商铺查询、缓存策略、附近商铺(GEO)")
@RestController
@RequestMapping("/shop")
@RequiredArgsConstructor
public class ShopController {

    private final IShopService shopService;

    /**
     * 查询商铺。
     * strategy: pass-through(默认) | mutex | logical
     */
    @Operation(summary = "查询商铺详情", description = "支持三种缓存策略:pass-through(旁路+空值防穿透)、mutex(互斥锁防击穿)、logical(逻辑过期防击穿)")
    @GetMapping("/{id}")
    public Result queryById(@PathVariable Long id,
                            @Parameter(description = "缓存策略: pass-through / mutex / logical")
                            @RequestParam(value = "strategy", defaultValue = "pass-through") String strategy) {
        return shopService.queryById(id, strategy);
    }

    @Operation(summary = "更新商铺", description = "先更新 DB 再删缓存(旁路缓存写策略)")
    @PutMapping
    public Result update(@RequestBody Shop shop) {
        return shopService.update(shop);
    }

    @Operation(summary = "按类型查商铺")
    @GetMapping("/of/type")
    public Result queryByType(@RequestParam("typeId") Long typeId,
                              @RequestParam(value = "current", defaultValue = "1") Integer current,
                              @RequestParam(value = "x", required = false) Double x,
                              @RequestParam(value = "y", required = false) Double y) {
        return shopService.queryByType(typeId, current, x, y);
    }

    /**
     * 附近商铺(GEO):按经纬度 + 距离 + 类型查
     */
    @Operation(summary = "附近商铺(GEO)", description = "用 Redis GEO + GeoSearch 按经纬度和半径查询")
    @GetMapping("/of/near")
    public Result queryNear(@RequestParam("typeId") Long typeId,
                            @RequestParam("x") Double x,
                            @RequestParam("y") Double y,
                            @RequestParam(value = "distKm", defaultValue = "5") Double distKm) {
        return shopService.queryShopByBiz(typeId, x, y, distKm);
    }

    /**
     * 商铺类型列表
     */
    @Operation(summary = "商铺类型列表")
    @GetMapping("/type/list")
    public Result queryTypeList() {
        return shopService.queryTypeList();
    }
}
