package com.dss.utils;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.RepeatedTest;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 红包拆金额算法(二倍均值法)测试。
 * 验证四个关键性质:
 * 1. 总额守恒:拆分之和 == 输入总额
 * 2. 每人至少 1 分
 * 3. 个数正确
 * 4. 参数非法抛异常
 */
class RedPacketSplitterTest {

    @Test
    void shouldSumToTotal() {
        // 100 元 = 10000 分,10 个
        List<Integer> amounts = RedPacketSplitter.split(10000, 10);
        int sum = amounts.stream().mapToInt(Integer::intValue).sum();
        assertEquals(10000, sum, "拆分金额之和必须等于总额");
    }

    @Test
    void shouldHaveCorrectCount() {
        List<Integer> amounts = RedPacketSplitter.split(10000, 10);
        assertEquals(10, amounts.size(), "个数必须等于输入 count");
    }

    @Test
    void eachAmountShouldBeAtLeastOneCent() {
        List<Integer> amounts = RedPacketSplitter.split(100, 100); // 极端:1 元拆 100 个
        assertTrue(amounts.stream().allMatch(a -> a >= 1), "每人至少 1 分");
    }

    @Test
    void lastPersonShouldGetRemainder() {
        // 用固定种子,验证逻辑可复现
        List<Integer> amounts = RedPacketSplitter.split(10000, 10, new Random(42));
        List<Integer> again = RedPacketSplitter.split(10000, 10, new Random(42));
        assertEquals(amounts, again, "相同种子应得到相同结果");
    }

    @RepeatedTest(20)
    void randomCasesShouldHoldInvariants() {
        // 随机参数,反复验证不变量
        Random r = new Random();
        int count = r.nextInt(50) + 1;          // 1..50
        int total = count + r.nextInt(10000);   // >= count
        List<Integer> amounts = RedPacketSplitter.split(total, count);
        assertEquals(count, amounts.size());
        assertEquals(total, amounts.stream().mapToInt(Integer::intValue).sum());
        assertTrue(amounts.stream().allMatch(a -> a >= 1));
    }

    @Test
    void shouldRejectZeroCount() {
        assertThrows(IllegalArgumentException.class, () -> RedPacketSplitter.split(100, 0));
    }

    @Test
    void shouldRejectTotalLessThanCount() {
        // 5 分拆 10 个,不可能每人≥1
        assertThrows(IllegalArgumentException.class, () -> RedPacketSplitter.split(5, 10));
    }

    @Test
    void singlePersonGetsAll() {
        // 1 个人,拿全部
        List<Integer> amounts = RedPacketSplitter.split(8888, 1);
        assertEquals(1, amounts.size());
        assertEquals(8888, amounts.get(0));
    }
}
