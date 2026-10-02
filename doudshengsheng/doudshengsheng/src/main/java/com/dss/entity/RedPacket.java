package com.dss.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("tb_red_packet")
public class RedPacket {
    @TableId(type = IdType.NONE) // 由 RedisIdWorker 生成
    private Long id;
    private String title;
    private Long totalAmount; // 分
    private Integer count;
    private Integer remainCount;
    private Integer gotCount;
    private Integer status; // 1进行中 2已抢完 3已退款关闭
    private LocalDateTime createTime;
    private LocalDateTime endTime;
}
