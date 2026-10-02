package com.dss.utils;

import com.dss.DoudShengShengApplication;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 全局唯一 ID 生成器测试。
 * 验证:单调递增、并发不重复、结构(时间戳位 + 序列号位)。
 */
@SpringBootTest(classes = DoudShengShengApplication.class)
class RedisIdWorkerTest {

    @Autowired
    private RedisIdWorker idWorker;

    @Autowired
    private StringRedisTemplate redis;

    @Test
    void shouldBeUniqueUnderConcurrency() throws InterruptedException {
        int threads = 100, perThread = 100;
        ExecutorService pool = Executors.newFixedThreadPool(16);
        CountDownLatch latch = new CountDownLatch(threads);
        Set<Long> ids = new HashSet<>();
        Object lock = new Object();

        for (int i = 0; i < threads; i++) {
            pool.submit(() -> {
                for (int j = 0; j < perThread; j++) {
                    long id = idWorker.nextId("test");
                    synchronized (lock) { ids.add(id); }
                }
                latch.countDown();
            });
        }
        latch.await();
        pool.shutdown();

        // 100 线程 × 100 次 = 10000 个 id,应全部唯一
        assertEquals(threads * perThread, ids.size(), "并发生成的 ID 必须全部唯一");
    }

    @Test
    void shouldBeMonotonic() {
        long a = idWorker.nextId("mono");
        long b = idWorker.nextId("mono");
        long c = idWorker.nextId("mono");
        assertTrue(b > a && c > b, "同一前缀的 ID 应单调递增");
    }

    @Test
    void shouldHaveTimestampInHighBits() {
        // 时间戳在高位,序列号在低 32 位。连续两个 id 高位应相同(同一秒),低位递增。
        long a = idWorker.nextId("struct");
        long b = idWorker.nextId("struct");
        long highA = a >>> 32;
        long highB = b >>> 32;
        long lowA = a & 0xFFFFFFFFL;
        long lowB = b & 0xFFFFFFFFL;
        assertTrue(lowB > lowA, "低 32 位序列号应递增");
        // 高位时间戳应相近(允许跨秒)
        assertTrue(Math.abs(highA - highB) <= 1, "高位时间戳应相同或相邻");
    }
}
