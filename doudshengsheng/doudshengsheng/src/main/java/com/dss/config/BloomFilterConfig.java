package com.dss.config;

import com.dss.entity.Shop;
import com.dss.mapper.ShopMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBloomFilter;
import org.redisson.api.RedissonClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 布隆过滤器配置:防缓存穿透。
 * <p>
 * 启动时把所有商铺 id 预热进布隆过滤器。
 * 查询前先过布隆:不存在则直接返回,不查缓存不查 DB。
 * <p>
 * 注意:布隆有误判率(说"存在"可能不存在),所以仍需空值缓存兜底误判的请求。
 * 布隆不支持删除,删商铺不能从过滤器移除(走空值缓存兜底)。
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
public class BloomFilterConfig {

    private final ShopMapper shopMapper;

    @Bean
    public RBloomFilter<Long> shopBloomFilter(RedissonClient redisson) {
        RBloomFilter<Long> filter = redisson.getBloomFilter("dss:bloom:shop");
        // 容量 10 万,误判率 1%(位图约 100KB)
        filter.tryInit(100_000L, 0.01);
        // 预热:把 DB 所有商铺 id 加入
        int count = 0;
        for (Shop shop : shopMapper.selectList(null)) {
            filter.add(shop.getId());
            count++;
        }
        log.info("布隆过滤器初始化完成,预热商铺 id {} 个", count);
        return filter;
    }
}
