package com.dss.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("tb_shop")
public class Shop {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String name;
    private Long typeId;
    private String images;
    private String cover; // 封面图URL
    private String area;
    private String address;
    private BigDecimal x;
    private BigDecimal y;
    private Long avgPrice;
    private Integer sold;
    private Integer comments;
    private Integer score;
    private String openHours;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
