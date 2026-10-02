package service

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"

	"github.com/redis/go-redis/v9"
)

type RedPacketService struct {
	grabScript    *redis.Script
	rateScript    *redis.Script
	consumerName  string
	once          sync.Once
	refundDelaySec int64 // 退款延迟秒
}

func NewRedPacketService() *RedPacketService {
	grabBytes, _ := os.ReadFile("resources/lua/redpacket_grab.lua")
	rateBytes, _ := os.ReadFile("resources/lua/sliding_window.lua")
	return &RedPacketService{
		grabScript:    redis.NewScript(string(grabBytes)),
		rateScript:    redis.NewScript(string(rateBytes)),
		consumerName:  "rp-c1-go",
		refundDelaySec: 120,
	}
}

// Create 创建红包雨:二倍均值法拆金额 + 预分配 List + 入延迟退款队列
func (s *RedPacketService) Create(ctx context.Context, title string, totalYuan, count int) *utils.Result {
	if totalYuan <= 0 || count <= 0 || totalYuan < count {
		return utils.Fail("参数非法")
	}
	totalCents := totalYuan * 100
	amounts := utils.SplitRedPacket(totalCents, count)
	id := utils.NextID("redpacket")

	// 预分配金额到 Redis List
	amountsKey := utils.RedpacketKey + fmt.Sprintf("%d", id) + utils.RedpacketAmounts
	strAmounts := make([]interface{}, len(amounts))
	for i, a := range amounts {
		strAmounts[i] = strconv.Itoa(a)
	}
	utils.Redis.RPush(ctx, amountsKey, strAmounts...)

	// 元数据 Hash
	metaKey := utils.RedpacketKey + fmt.Sprintf("%d", id) + utils.RedpacketMeta
	ttl := time.Duration(s.refundDelaySec+60) * time.Second
	utils.Redis.HSet(ctx, metaKey, map[string]interface{}{
		"total":    strconv.Itoa(totalCents),
		"count":    strconv.Itoa(count),
		"remain":   strconv.Itoa(count),
		"got":      "0",
		"status":   "1",
		"expireAt": strconv.FormatInt(time.Now().UnixMilli()+s.refundDelaySec*1000, 10),
	})
	utils.Redis.Expire(ctx, amountsKey, ttl)
	utils.Redis.Expire(ctx, metaKey, ttl)
	utils.Redis.Expire(ctx, utils.RedpacketKey+fmt.Sprintf("%d", id)+utils.RedpacketTaken, ttl)

	// 入延迟退款队列(goroutine + time.After,对应 Java 的 ScheduledExecutor)
	go func() {
		time.Sleep(time.Duration(s.refundDelaySec) * time.Second)
		s.refundUnclaimed(utils.Ctx, id)
	}()

	// 落库
	rp := model.RedPacket{
		ID: id, Title: title, TotalAmount: int64(totalCents),
		Count: count, RemainCount: count, GotCount: 0, Status: 1,
		CreatedAt: time.Now(),
	}
	utils.DB.Create(&rp)
	return utils.OKWith(id)
}

// Grab 抢红包:滑动窗口限流 + Lua 原子抢 + 异步落库
func (s *RedPacketService) Grab(ctx context.Context, rpID, userID int64) *utils.Result {
	// 1. 滑动窗口限流(10秒3次)
	rateKey := utils.RedpacketRateKey + fmt.Sprintf("%d:%d", rpID, userID)
	member := fmt.Sprintf("%d:%d", time.Now().UnixMilli(), userID)
	allowed, err := s.rateScript.Run(ctx, utils.Redis, []string{rateKey},
		strconv.FormatInt(time.Now().UnixMilli(), 10),
		"10000", "3", member,
	).Int()
	if err == nil && allowed == 0 {
		return utils.FailWith(429, "操作太频繁,请稍后再试")
	}

	// 2. Lua 原子抢
	idStr := strconv.FormatInt(rpID, 10)
	keys := []string{
		utils.RedpacketKey + idStr + utils.RedpacketAmounts,
		utils.RedpacketKey + idStr + utils.RedpacketTaken,
		utils.RedpacketKey + idStr + utils.RedpacketMeta,
	}
	result, err := s.grabScript.Run(ctx, utils.Redis, keys,
		strconv.FormatInt(userID, 10),
		strconv.FormatInt(time.Now().Unix(), 10),
	).Text()
	if err != nil {
		return utils.Fail("红包异常")
	}
	if result == "-1" {
		return utils.Fail("您已领过该红包")
	}
	if result == "0" {
		return utils.Fail("红包已抢完")
	}
	amount, _ := strconv.ParseInt(result, 10, 64)
	// 3. 异步落库 Stream
	utils.Redis.XAdd(ctx, &redis.XAddArgs{
		Stream: utils.RedpacketRecordStream,
		Values: map[string]interface{}{
			"redPacketId": idStr,
			"userId":      strconv.FormatInt(userID, 10),
			"amount":      result,
		},
	})
	return utils.OKWith(amount)
}

// Rank 排行榜(按金额降序)
func (s *RedPacketService) Rank(ctx context.Context, rpID int64) *utils.Result {
	takenKey := utils.RedpacketKey + fmt.Sprintf("%d", rpID) + utils.RedpacketTaken
	taken, _ := utils.Redis.HGetAll(ctx, takenKey).Result()
	type rec struct {
		UserID string `json:"userId"`
		Amount string `json:"amount"`
	}
	list := []rec{}
	for k, v := range taken {
		if len(k) > 5 && k[len(k)-5:] == ":time" {
			continue // 跳过 :time 字段
		}
		list = append(list, rec{UserID: k, Amount: v})
	}
	// 按金额降序
	for i := 0; i < len(list); i++ {
		for j := i + 1; j < len(list); j++ {
			ai, _ := strconv.ParseInt(list[i].Amount, 10, 64)
			aj, _ := strconv.ParseInt(list[j].Amount, 10, 64)
			if aj > ai {
				list[i], list[j] = list[j], list[i]
			}
		}
	}
	return utils.OKWith(list)
}

// StartConsumer 启动红包记录消费者
func (s *RedPacketService) StartConsumer() {
	s.once.Do(func() {
		go s.consumeLoop()
	})
}

func (s *RedPacketService) consumeLoop() {
	ctx := utils.Ctx
	utils.Redis.XGroupCreateMkStream(ctx, utils.RedpacketRecordStream, "g1", "$")
	for {
		streams, err := utils.Redis.XReadGroup(ctx, &redis.XReadGroupArgs{
			Group: "g1", Consumer: s.consumerName,
			Streams: []string{utils.RedpacketRecordStream, ">"},
			Count:   50, Block: 2 * time.Second,
		}).Result()
		if err != nil || len(streams) == 0 {
			time.Sleep(time.Second)
			continue
		}
		for _, stream := range streams {
			for _, msg := range stream.Messages {
				s.handleRecord(ctx, msg)
			}
		}
	}
}

func (s *RedPacketService) handleRecord(ctx context.Context, msg redis.XMessage) {
	defer utils.Redis.XAck(ctx, utils.RedpacketRecordStream, "g1", msg.ID)
	rpID, _ := toInt64(msg.Values["redPacketId"])
	uid, _ := toInt64(msg.Values["userId"])
	amount, _ := toInt64(msg.Values["amount"])
	utils.DB.Create(&model.RedPacketRecord{
		RedPacketID: rpID, UserID: uid, Amount: amount, GrabTime: time.Now(),
	})
}

// refundUnclaimed 未领退款(到期扫描)
func (s *RedPacketService) refundUnclaimed(ctx context.Context, rpID int64) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("退款异常:", r)
		}
	}()
	metaKey := utils.RedpacketKey + fmt.Sprintf("%d", rpID) + utils.RedpacketMeta
	amountsKey := utils.RedpacketKey + fmt.Sprintf("%d", rpID) + utils.RedpacketAmounts
	remainStr, _ := utils.Redis.HGet(ctx, metaKey, "remain").Result()
	remain, _ := strconv.Atoi(remainStr)
	if remain > 0 {
		leftAmounts, _ := utils.Redis.LRange(ctx, amountsKey, 0, -1).Result()
		refund := int64(0)
		for _, a := range leftAmounts {
			v, _ := strconv.ParseInt(a, 10, 64)
			refund += v
		}
		fmt.Printf("红包 %d 未领退款:剩余 %d 个,金额 %d 分\n", rpID, remain, refund)
		utils.Redis.HSet(ctx, metaKey, "status", "3")
		var rp model.RedPacket
		if utils.DB.First(&rp, rpID).Error == nil {
			now := time.Now()
			rp.Status = 3
			rp.RemainCount = remain
			rp.EndTime = &now
			utils.DB.Save(&rp)
		}
	}
}

// ListRedPackets 后台:场次列表
func (s *RedPacketService) ListRedPackets(ctx context.Context) *utils.Result {
	var list []model.RedPacket
	utils.DB.Order("create_time desc").Limit(50).Find(&list)
	return utils.OKWith(list)
}

// RedPacketDetail 后台:场次详情
func (s *RedPacketService) RedPacketDetail(ctx context.Context, rpID int64) *utils.Result {
	var rp model.RedPacket
	if err := utils.DB.First(&rp, rpID).Error; err != nil {
		return utils.Fail("场次不存在")
	}
	metaKey := utils.RedpacketKey + fmt.Sprintf("%d", rpID) + utils.RedpacketMeta
	meta, _ := utils.Redis.HGetAll(ctx, metaKey).Result()
	takenKey := utils.RedpacketKey + fmt.Sprintf("%d", rpID) + utils.RedpacketTaken
	takenCount, _ := utils.Redis.HLen(ctx, takenKey).Result()
	return utils.OKWith(map[string]interface{}{
		"info":       rp,
		"meta":       meta,
		"takenCount": takenCount / 2, // 每人有 amount+time 两个 field
	})
}
