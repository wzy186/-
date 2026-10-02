package com.dss.service.impl;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.dto.Result;
import com.dss.entity.Shop;
import com.dss.entity.ShopType;
import com.dss.mapper.ShopMapper;
import com.dss.mapper.ShopTypeMapper;
import com.dss.service.IShopService;
import com.dss.utils.CacheClient;
import com.dss.utils.RedisConstants;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.geo.Distance;
import org.springframework.data.geo.GeoResult;
import org.springframework.data.geo.GeoResults;
import org.springframework.data.geo.Metrics;
import org.springframework.data.geo.Point;
import org.springframework.data.redis.connection.RedisGeoCommands;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.domain.geo.GeoReference;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static com.dss.utils.RedisConstants.CACHE_SHOP_TTL;

@Slf4j
@Service
@RequiredArgsConstructor
public class ShopServiceImpl implements IShopService {

    private final ShopMapper shopMapper;
    private final ShopTypeMapper shopTypeMapper;
    private final CacheClient cacheClient;
    private final StringRedisTemplate redis;
    private final org.redisson.api.RBloomFilter<Long> shopBloomFilter;

    /**
     * 查询商铺。
     * strategy: pass-through(默认) | mutex | logical
     * 体现缓存三大问题的不同解法。
     */
    @Override
    public Result queryById(Long id, String strategy) {
        // 布隆过滤器前置判断:不存在则直接返回,不查缓存不查 DB
        // (布隆说"不存在"一定不存在,挡掉 99% 恶意穿透;误判的 1% 走空值缓存兜底)
        if (!shopBloomFilter.contains(id)) {
            return Result.fail("商铺不存在");
        }
        Shop shop;
        if ("mutex".equals(strategy)) {
            shop = cacheClient.queryWithMutex(
                    RedisConstants.CACHE_SHOP_KEY, id, Shop.class,
                    this::getById, CACHE_SHOP_TTL, TimeUnit.MINUTES);
        } else if ("logical".equals(strategy)) {
            // 逻辑过期方案需预热:首次未命中时,主动用逻辑过期写入,再查一次
            shop = cacheClient.queryWithLogicalExpire(
                    RedisConstants.CACHE_SHOP_LOGIC_KEY, id, Shop.class,
                    this::getById, CACHE_SHOP_TTL, TimeUnit.MINUTES);
            if (shop == null) {
                Shop dbShop = getById(id);
                if (dbShop != null) {
                    cacheClient.setWithLogicalExpire(
                            RedisConstants.CACHE_SHOP_LOGIC_KEY + id, dbShop, CACHE_SHOP_TTL, TimeUnit.MINUTES);
                    shop = dbShop;
                }
            }
        } else {
            // 默认旁路缓存 + 空值缓存防穿透 + 随机 TTL 防雪崩
            shop = cacheClient.queryWithPassThrough(
                    RedisConstants.CACHE_SHOP_KEY, id, Shop.class,
                    this::getById, CACHE_SHOP_TTL + (long)(Math.random() * 10), TimeUnit.MINUTES);
        }
        if (shop == null) {
            return Result.fail("商铺不存在");
        }
        return Result.ok(shop);
    }

    private Shop getById(Long id) {
        return shopMapper.selectById(id);
    }

    /**
     * 更新商铺:先更新 DB,再删缓存(旁路缓存写策略)
     */
    @Override
    public Result update(Shop shop) {
        if (shop.getId() == null) {
            return Result.fail("商铺 id 不能为空");
        }
        // 1. 更新 DB
        shopMapper.updateById(shop);
        // 2. 删除缓存(不能先删后更,会有并发脏读)
        redis.delete(RedisConstants.CACHE_SHOP_KEY + shop.getId());
        redis.delete(RedisConstants.CACHE_SHOP_LOGIC_KEY + shop.getId());
        return Result.ok();
    }

    /**
     * 新增商铺(管理员):写 DB + 加布隆过滤器(否则查不到)+ 加 GEO
     */
    @Override
    public Result save(Shop shop) {
        if (shop.getName() == null || shop.getTypeId() == null) {
            return Result.fail("商铺名和类型不能为空");
        }
        shopMapper.insert(shop);
        // 加入布隆过滤器,否则新增的商铺会被布隆挡住查不到
        shopBloomFilter.add(shop.getId());
        // 有坐标则加入 GEO
        if (shop.getX() != null && shop.getY() != null) {
            redis.opsForGeo().add(RedisConstants.SHOP_GEO_KEY,
                    new org.springframework.data.geo.Point(shop.getX().doubleValue(), shop.getY().doubleValue()),
                    shop.getId().toString());
        }
        return Result.ok(shop.getId());
    }

    @Override
    public Result queryByType(Long typeId, Integer current, Double x, Double y) {
        // 简化:直接查 DB,GEO 在 ShopController 走 Redis
        List<Shop> shops = shopMapper.selectList(
                new LambdaQueryWrapper<Shop>().eq(Shop::getTypeId, typeId));
        return Result.ok(shops);
    }

    /**
     * 附近商铺:用 GEO 数据结构按经纬度 + 距离查询。
     * 流程:GEOSEARCH 取 shopId + 距离 → 按 typeId 过滤 → 批量查 DB 拼距离。
     */
    @Override
    public Result queryShopByBiz(Long typeId, Double x, Double y, Double distKm) {
        if (x == null || y == null) {
            return queryByType(typeId, 1, x, y);
        }
        Point point = new Point(x, y);
        Distance radius = new Distance(distKm, Metrics.KILOMETERS);
        RedisGeoCommands.GeoSearchCommandArgs args =
                RedisGeoCommands.GeoSearchCommandArgs.newGeoSearchArgs()
                        .includeDistance()
                        .sortAscending()
                        .limit(10);
        GeoReference<String> ref = GeoReference.fromCoordinate(point);
        GeoResults<RedisGeoCommands.GeoLocation<String>> results =
                redis.opsForGeo().search(RedisConstants.SHOP_GEO_KEY, ref, radius, args);
        if (results == null || results.getContent().isEmpty()) {
            return Result.ok(java.util.Collections.emptyList());
        }
        List<Long> ids = new ArrayList<>();
        Map<Long, Double> distMap = new HashMap<>();
        for (GeoResult<RedisGeoCommands.GeoLocation<String>> r : results) {
            String shopId = r.getContent().getName();
            if (shopId == null) continue;
            Long id = Long.valueOf(shopId);
            ids.add(id);
            distMap.put(id, r.getDistance().getValue());
        }
        List<Shop> shops = shopMapper.selectBatchIds(ids);
        List<Shop> filtered = shops.stream()
                .filter(s -> typeId == null || typeId.equals(s.getTypeId()))
                .collect(Collectors.toList());
        Map<String, Object> resp = new HashMap<>();
        resp.put("shops", filtered);
        resp.put("distances", distMap);
        return Result.ok(resp);
    }

    /**
     * 商铺类型列表,用 Redis String 存整个 JSON 列表(列表型缓存)
     */
    @Override
    public Result queryTypeList() {
        String key = RedisConstants.CACHE_SHOP_KEY + "type:list";
        String json = redis.opsForValue().get(key);
        if (json != null && !json.isEmpty()) {
            return Result.ok(JSONUtil.toList(json, ShopType.class));
        }
        List<ShopType> types = shopTypeMapper.selectList(null);
        redis.opsForValue().set(key, JSONUtil.toJsonStr(types), CACHE_SHOP_TTL, TimeUnit.MINUTES);
        return Result.ok(types);
    }
}
