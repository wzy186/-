package com.dss.controller;

import com.dss.dto.Result;
import com.dss.service.IRedPacketService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Tag(name = "红包雨", description = "红包雨创建、抢红包、领取排行榜")
@RestController
@RequestMapping("/redpacket")
@RequiredArgsConstructor
public class RedPacketController {

    private final IRedPacketService redPacketService;

    /**
     * 创建红包雨场次
     */
    @Operation(summary = "创建红包雨场次", description = "二倍均值法拆金额,预分配到 Redis List")
    @PostMapping("/create")
    public Result create(@Parameter(description = "场次标题") @RequestParam("title") String title,
                         @Parameter(description = "总金额(元)") @RequestParam("totalYuan") int totalYuan,
                         @Parameter(description = "红包个数") @RequestParam("count") int count) {
        return redPacketService.createRedPacket(title, totalYuan, count);
    }

    /**
     * 抢红包
     */
    @Operation(summary = "抢红包", description = "Lua 原子抢红包 + 用户级滑动窗口限流防刷 + 异步落库")
    // 注:抢红包不用 IP 级限流——同一网络下多用户抢红包是正常场景,IP 限流会误伤。
    // 防刷靠用户级滑动窗口(sliding_window.lua,10秒3次)+ Lua 幂等(一人一次)。
    @PostMapping("/grab/{id}")
    public Result grab(@Parameter(description = "红包场次 ID") @PathVariable("id") Long redPacketId) {
        return redPacketService.grab(redPacketId);
    }

    /**
     * 领取排行榜(按金额降序)
     */
    @Operation(summary = "领取排行榜", description = "按领取金额降序排列")
    @GetMapping("/rank/{id}")
    public Result rank(@Parameter(description = "红包场次 ID") @PathVariable("id") Long redPacketId) {
        return redPacketService.rank(redPacketId);
    }
}
