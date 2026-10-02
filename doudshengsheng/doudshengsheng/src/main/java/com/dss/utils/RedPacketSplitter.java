package com.dss.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 红包金额拆分算法:二倍均值法。
 * <p>
 * 每人金额 ∈ [1, 剩余金额 / 剩余人数 × 2 - 1],最后一人拿剩余。
 * 性质:
 * - 每人至少 1 分
 * - 任意抢的顺序下,每个人金额的数学期望相等(公平)
 * - 总额守恒:拆分金额之和 == totalCents
 */
public final class RedPacketSplitter {

    private RedPacketSplitter() {}

    /**
     * 拆分金额(单位:分)。
     *
     * @param totalCents 总金额(分),必须 >= count
     * @param count      个数,必须 > 0
     * @return 拆分后的金额列表(打乱顺序),长度 == count
     * @throws IllegalArgumentException 参数非法
     */
    public static List<Integer> split(int totalCents, int count) {
        return split(totalCents, count, new Random());
    }

    /**
     * 拆分金额,可传入 Random(测试用固定种子可复现)。
     */
    public static List<Integer> split(int totalCents, int count, Random random) {
        if (count <= 0) {
            throw new IllegalArgumentException("个数必须大于 0");
        }
        if (totalCents < count) {
            throw new IllegalArgumentException("总金额(分)不能小于个数");
        }
        List<Integer> amounts = new ArrayList<>(count);
        int remain = totalCents;
        int remainPeople = count;
        for (int i = 0; i < count - 1; i++) {
            int max = remain / remainPeople * 2 - 1;
            if (max < 1) max = 1;
            int amt = random.nextInt(max) + 1; // [1, max]
            amounts.add(amt);
            remain -= amt;
            remainPeople--;
        }
        amounts.add(remain); // 最后一人拿剩余
        Collections.shuffle(amounts, random);
        return amounts;
    }
}
