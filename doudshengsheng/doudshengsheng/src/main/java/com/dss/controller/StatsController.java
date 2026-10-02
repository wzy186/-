package com.dss.controller;

import com.dss.dto.Result;
import com.dss.service.IStatsService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/stats")
@RequiredArgsConstructor
public class StatsController {

    private final IStatsService statsService;

    @PostMapping("/sign")
    public Result sign() {
        return statsService.sign();
    }

    @GetMapping("/sign/count")
    public Result signCount() {
        return statsService.signCount();
    }

    @GetMapping("/sign/records")
    public Result signRecords() {
        return statsService.signRecords();
    }

    @PostMapping("/uv")
    public Result uv(@RequestParam("bizKey") String bizKey,
                     @RequestParam(value = "userId", required = false) Long userId) {
        return statsService.uv(bizKey, userId);
    }

    @GetMapping("/uv/count")
    public Result uvCount(@RequestParam("bizKey") String bizKey) {
        return statsService.uvCount(bizKey);
    }
}
