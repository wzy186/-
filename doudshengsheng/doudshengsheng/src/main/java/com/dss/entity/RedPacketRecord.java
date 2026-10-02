package com.dss.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("tb_red_packet_record")
public class RedPacketRecord {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long redPacketId;
    private Long userId;
    private Long amount; // 分
    private LocalDateTime grabTime;
}
