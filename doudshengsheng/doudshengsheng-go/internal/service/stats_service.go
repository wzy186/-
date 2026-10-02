package service

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"doudshengsheng-go/internal/utils"
)

type StatsService struct{}

// Sign 签到 BitMap
func (s *StatsService) Sign(ctx context.Context, userID int64) *utils.Result {
	now := time.Now()
	key := utils.SignKey + fmt.Sprintf("%d:%s", userID, now.Format("200601"))
	utils.Redis.SetBit(ctx, key, int64(now.Day()-1), 1)
	return utils.OK()
}

// SignCount 连续签到天数
func (s *StatsService) SignCount(ctx context.Context, userID int64) *utils.Result {
	now := time.Now()
	key := utils.SignKey + fmt.Sprintf("%d:%s", userID, now.Format("200601"))
	// 从今天往前数连续 1
	count := 0
	for d := now.Day(); d >= 1; d-- {
		bit, _ := utils.Redis.GetBit(ctx, key, int64(d-1)).Result()
		if bit == 1 {
			count++
		} else {
			break
		}
	}
	return utils.OKWith(count)
}

// SignRecords 本月签到记录(每天的布尔数组,下标=日期)
func (s *StatsService) SignRecords(ctx context.Context, userID int64) *utils.Result {
	now := time.Now()
	key := utils.SignKey + fmt.Sprintf("%d:%s", userID, now.Format("200601"))
	records := []bool{false} // 下标0占位
	for d := 1; d <= now.Day(); d++ {
		bit, _ := utils.Redis.GetBit(ctx, key, int64(d-1)).Result()
		records = append(records, bit == 1)
	}
	return utils.OKWith(records)
}

// UV 记录访客(HyperLogLog)
func (s *StatsService) UV(ctx context.Context, bizKey string, userID int64) *utils.Result {
	utils.Redis.PFAdd(ctx, utils.UVKey+bizKey, strconv.FormatInt(userID, 10))
	return utils.OK()
}

// UVCount UV 数
func (s *StatsService) UVCount(ctx context.Context, bizKey string) *utils.Result {
	cnt, _ := utils.Redis.PFCount(ctx, utils.UVKey+bizKey).Result()
	return utils.OKWith(cnt)
}
