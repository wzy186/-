package utils

import (
	"fmt"
	"time"
)

// 2024-01-01 的秒级时间戳,作为 ID 的高位基准
var beginTimestamp = time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).Unix()

// NextID 全局唯一 ID:符号位(1) + 时间戳(31) + 序列号(32)
// 对应 Java 的 RedisIdWorker,时间戳左移 32 位,低位放 Redis 自增序列号
func NextID(keyPrefix string) int64 {
	nowSecond := time.Now().Unix()
	timestamp := nowSecond - beginTimestamp
	date := time.Now().Format("20060102")
	count, err := Redis.Incr(Ctx, IDSeqKey+keyPrefix+":"+date).Result()
	if err != nil {
		fmt.Println("ID 自增失败:", err)
		return 0
	}
	return (timestamp << 32) | count
}
