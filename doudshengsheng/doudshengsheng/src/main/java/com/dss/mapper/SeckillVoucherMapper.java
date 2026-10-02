package com.dss.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.dss.entity.SeckillVoucher;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Update;

public interface SeckillVoucherMapper extends BaseMapper<SeckillVoucher> {

    /**
     * 乐观锁扣减库存:WHERE stock > 0 保证并发下不超卖(CAS 语义)。
     * 返回受影响行数:1=扣减成功,0=库存已不足(扣减失败)。
     */
    @Update("UPDATE tb_seckill_voucher SET stock = stock - 1 " +
            "WHERE voucher_id = #{voucherId} AND stock > 0")
    int deductStock(@Param("voucherId") Long voucherId);
}
