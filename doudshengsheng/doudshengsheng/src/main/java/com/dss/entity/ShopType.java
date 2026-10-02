package com.dss.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_shop_type")
public class ShopType {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String name;
    private Integer sort;
}
