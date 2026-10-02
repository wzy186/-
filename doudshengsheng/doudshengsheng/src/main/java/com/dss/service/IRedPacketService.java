package com.dss.service;

import com.dss.dto.Result;

public interface IRedPacketService {
    /** 创建红包雨场次:二倍均值法拆金额 + 预分配到 List + 入延迟退款队列 */
    Result createRedPacket(String title, int totalYuan, int count);
    /** 抢红包:限流 + 幂等 + Lua 原子取金额 + 异步落库 */
    Result grab(Long redPacketId);
    /** 查询领取记录排行榜(按金额降序) */
    Result rank(Long redPacketId);
    /** 后台:红包雨场次列表 */
    Result listRedPackets();
    /** 后台:场次实时详情(含 Redis 里的领取进度) */
    Result redPacketDetail(Long redPacketId);
}
