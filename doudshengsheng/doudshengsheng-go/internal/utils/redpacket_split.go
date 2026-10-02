package utils

import "math/rand"

// SplitRedPacket 二倍均值法拆红包金额(单位:分)
// 每人金额 ∈ [1, 剩余金额/剩余人数×2 - 1],最后一人拿剩余
// 性质:每人至少1分、总额守恒、期望公平
func SplitRedPacket(totalCents, count int) []int {
	if count <= 0 {
		panic("个数必须大于0")
	}
	if totalCents < count {
		panic("总金额不能小于个数")
	}
	amounts := make([]int, 0, count)
	remain := totalCents
	remainPeople := count
	for i := 0; i < count-1; i++ {
		max := remain/remainPeople*2 - 1
		if max < 1 {
			max = 1
		}
		amt := rand.Intn(max) + 1 // [1, max]
		amounts = append(amounts, amt)
		remain -= amt
		remainPeople--
	}
	amounts = append(amounts, remain) // 最后一人
	// 打乱
	rand.Shuffle(len(amounts), func(i, j int) {
		amounts[i], amounts[j] = amounts[j], amounts[i]
	})
	return amounts
}
