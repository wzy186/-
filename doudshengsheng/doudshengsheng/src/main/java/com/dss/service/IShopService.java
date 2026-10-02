package com.dss.service;

import com.dss.dto.Result;
import com.dss.entity.Shop;

public interface IShopService {
    Result queryById(Long id, String strategy);
    Result update(Shop shop);
    /** 新增商铺(管理员):写 DB + 加布隆过滤器 + 加 GEO */
    Result save(Shop shop);
    Result queryByType(Long typeId, Integer current, Double x, Double y);
    Result queryTypeList();
    /** 附近商铺:GEO 按经纬度 + 距离查询 */
    Result queryShopByBiz(Long typeId, Double x, Double y, Double distKm);
}
