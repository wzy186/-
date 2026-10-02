package com.dss.config;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.entity.Blog;
import com.dss.entity.SeckillVoucher;
import com.dss.entity.Shop;
import com.dss.mapper.BlogMapper;
import com.dss.mapper.SeckillVoucherMapper;
import com.dss.mapper.ShopMapper;
import com.dss.service.IRedPacketService;
import com.dss.ai.rag.RagService;
import com.dss.utils.RedisConstants;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;
import org.springframework.data.geo.Point;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.List;

/**
 * 启动预热:
 * 1. 秒杀券库存 → Redis
 * 2. 商铺坐标 → GEO
 * 3. 笔记点赞数 → Redis ZSet(让点赞榜有初始数据)
 * 4. 若无进行中红包雨,自动创建一个(让首页有红包可抢)
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
public class DataInitializer {

    private final SeckillVoucherMapper seckillVoucherMapper;
    private final ShopMapper shopMapper;
    private final BlogMapper blogMapper;
    private final StringRedisTemplate redis;
    private final @Lazy IRedPacketService redPacketService;
    private final @Lazy RagService ragService;

    @Bean
    public ApplicationRunner preheat() {
        return args -> {
            // 1. 预热所有秒杀券库存
            List<SeckillVoucher> svs = seckillVoucherMapper.selectList(null);
            for (SeckillVoucher sv : svs) {
                redis.opsForValue().set(RedisConstants.SECKILL_STOCK_KEY + sv.getVoucherId(),
                        String.valueOf(sv.getStock()));
            }
            log.info("秒杀券库存预热完成,共 {} 张", svs.size());

            // 2. 预热商铺坐标到 GEO
            List<Shop> shops = shopMapper.selectList(
                    new LambdaQueryWrapper<Shop>().isNotNull(Shop::getX).isNotNull(Shop::getY));
            for (Shop shop : shops) {
                redis.opsForGeo().add(RedisConstants.SHOP_GEO_KEY,
                        new Point(shop.getX().doubleValue(), shop.getY().doubleValue()),
                        shop.getId().toString());
            }
            log.info("商铺 GEO 预热完成,共 {} 家", shops.size());

            // 3. 预热笔记点赞榜:给每篇笔记造几个初始点赞(ZSet,用递增时间戳)
            List<Blog> blogs = blogMapper.selectList(null);
            for (Blog blog : blogs) {
                String key = RedisConstants.BLOG_LIKED_KEY + blog.getId();
                if (redis.opsForZSet().zCard(key) == 0 && blog.getLiked() != null && blog.getLiked() > 0) {
                    // 用 liked 数的一半造初始点赞用户(用户 2..n),时间戳递增
                    int n = Math.min(blog.getLiked() / 2, 10);
                    for (int i = 0; i < n; i++) {
                        long uid = 2 + i;
                        redis.opsForZSet().add(key, String.valueOf(uid), System.currentTimeMillis() + i);
                    }
                }
            }
            log.info("笔记点赞预热完成,共 {} 篇", blogs.size());

            // 4. 若无进行中红包雨场次,自动创建一个(50元10个,演示用)
            com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<com.dss.entity.RedPacket> qw =
                    new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<>();
            qw.eq("status", 1);
            Long activeCount = null;
            try {
                activeCount = redPacketService.listRedPackets().getData() != null
                        ? ((List<?>) redPacketService.listRedPackets().getData()).stream()
                            .filter(o -> Integer.valueOf(1).equals(((com.dss.entity.RedPacket) o).getStatus())).count()
                        : 0;
            } catch (Exception e) {
                log.warn("检查红包雨场次失败: {}", e.getMessage());
            }
            if (activeCount == null || activeCount == 0) {
                try {
                    redPacketService.createRedPacket("每日红包雨", 50, 10);
                    log.info("已自动创建演示红包雨场次(50元/10个)");
                } catch (Exception e) {
                    log.warn("自动创建红包雨失败: {}", e.getMessage());
                }
            }

            // 5. RAG 向量索引(异步,失败不影响启动;需 ollama bge-m3)
            new Thread(() -> {
                try {
                    int n = ragService.reindex();
                    log.info("RAG 向量索引完成,共 {} 条", n);
                } catch (Exception e) {
                    log.warn("RAG 索引失败(ollama 未就绪?): {}", e.getMessage());
                }
            }, "rag-reindex").start();
        };
    }
}
