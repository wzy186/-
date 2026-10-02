package com.dss.service;

import com.dss.dto.Result;

public interface IStatsService {
    /** 签到:BitMap SETBIT */
    Result sign();
    /** 连续签到天数 */
    Result signCount();
    /** 本月签到记录(每天的布尔数组,前端日历用) */
    Result signRecords();
    /** UV 统计:HyperLogLog 记录访客 */
    Result uv(String bizKey, Long userId);
    /** 查 UV 数 */
    Result uvCount(String bizKey);
}
