package service

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"

	"github.com/redis/go-redis/v9"
)

type SeckillService struct {
	seckillScript *redis.Script
	orderScript   *redis.Script
	consumerName  string
	once          sync.Once
}

func NewSeckillService() *SeckillService {
	// 加载 Lua 脚本(复用 Java 版 seckill.lua)
	scriptBytes, _ := os.ReadFile("resources/lua/seckill.lua")
	return &SeckillService{
		seckillScript: redis.NewScript(string(scriptBytes)),
		consumerName:  "c1-go",
	}
}

// Seckill 秒杀下单:Lua 原子校验+扣库存 → 发 Stream → 返回
func (s *SeckillService) Seckill(ctx context.Context, voucherID, userID int64) *utils.Result {
	// 查秒杀券信息(取时间)
	var sv model.SeckillVoucher
	if err := utils.DB.First(&sv, voucherID).Error; err != nil {
		return utils.Fail("秒杀券不存在")
	}
	now := time.Now().UnixMilli()
	// Lua 原子:时间+库存+一人一单+扣减
	keys := []string{
		utils.SeckillStockKey + fmt.Sprintf("%d", voucherID),
		utils.SeckillOrderKey + fmt.Sprintf("%d", voucherID),
	}
	res, err := s.seckillScript.Run(ctx, utils.Redis, keys,
		fmt.Sprintf("%d", userID),
		fmt.Sprintf("%d", now),
		fmt.Sprintf("%d", sv.BeginTime.UnixMilli()),
		fmt.Sprintf("%d", sv.EndTime.UnixMilli()),
	).Int()
	if err != nil {
		return utils.Fail("秒杀异常: " + err.Error())
	}
	if res != 0 {
		return utils.Fail(codeMsg(res))
	}
	// 校验通过,生成订单 ID,发 Stream 异步落库
	orderID := utils.NextID("order")
	msg := map[string]interface{}{
		"orderId":   fmt.Sprintf("%d", orderID),
		"userId":    fmt.Sprintf("%d", userID),
		"voucherId": fmt.Sprintf("%d", voucherID),
	}
	utils.Redis.XAdd(ctx, &redis.XAddArgs{
		Stream: utils.SeckillStreamKey,
		Values: msg,
	})
	return utils.OKWith(orderID)
}

// StartConsumer 启动 Stream 消费者(goroutine,对应 Java 的线程池消费)
func (s *SeckillService) StartConsumer() {
	s.once.Do(func() {
		go s.consumeLoop()
	})
}

func (s *SeckillService) consumeLoop() {
	ctx := utils.Ctx
	// 创建消费组(已存在则忽略)
	utils.Redis.XGroupCreateMkStream(ctx, utils.SeckillStreamKey, utils.SeckillStreamGroup, "$")
	for {
		err := s.consumeOnce(ctx)
		if err != nil {
			// 退避,避免异常死循环(对应 Java 修过的坑)
			time.Sleep(time.Second)
		}
	}
}

func (s *SeckillService) consumeOnce(ctx context.Context) error {
	streams, err := utils.Redis.XReadGroup(ctx, &redis.XReadGroupArgs{
		Group:    utils.SeckillStreamGroup,
		Consumer: s.consumerName,
		Streams:  []string{utils.SeckillStreamKey, ">"},
		Count:    1,
		Block:    2 * time.Second,
	}).Result()
	if err != nil || len(streams) == 0 {
		return nil
	}
	for _, stream := range streams {
		for _, msg := range stream.Messages {
			s.handleRecord(ctx, msg)
		}
	}
	return nil
}

// handleRecord 处理一条消息:分布式锁兜底 + 落库 + ACK
func (s *SeckillService) handleRecord(ctx context.Context, msg redis.XMessage) {
	defer func() {
		utils.Redis.XAck(ctx, utils.SeckillStreamKey, utils.SeckillStreamGroup, msg.ID)
	}()
	uid, _ := toInt64(msg.Values["userId"])
	vid, _ := toInt64(msg.Values["voucherId"])
	oid, _ := toInt64(msg.Values["orderId"])

	// 分布式锁兜底一人一单(go-redis 的 SetNX 简易锁,生产用 Redisson)
	lockKey := fmt.Sprintf("dss-go:lock:order:%d", uid)
	locked, _ := utils.Redis.SetNX(ctx, lockKey, "1", 10*time.Second).Result()
	if !locked {
		return
	}
	defer utils.Redis.Del(ctx, lockKey)

	// 落库
	order := model.VoucherOrder{
		ID:        oid,
		UserID:    uid,
		VoucherID: vid,
		PayType:   1,
		Status:    2,
		CreatedAt: time.Now(),
	}
	utils.DB.Create(&order)
}

// MyOrders 我的订单
func (s *SeckillService) MyOrders(ctx context.Context, userID int64) *utils.Result {
	var orders []model.VoucherOrder
	utils.DB.Where("user_id = ?", userID).Order("create_time desc").Find(&orders)
	return utils.OKWith(orders)
}

// OrderStats 订单统计(后台)
func (s *SeckillService) OrderStats(ctx context.Context) *utils.Result {
	var total int64
	utils.DB.Model(&model.VoucherOrder{}).Count(&total)
	type groupCnt struct {
		VoucherID int64 `gorm:"column:voucher_id" json:"voucherId"`
		Cnt       int64 `gorm:"column:cnt" json:"cnt"`
	}
	var byVoucher []groupCnt
	utils.DB.Model(&model.VoucherOrder{}).Select("voucher_id, count(*) as cnt").Group("voucher_id").Scan(&byVoucher)
	return utils.OKWith(map[string]interface{}{
		"total":     total,
		"byVoucher": byVoucher,
	})
}

func codeMsg(code int) string {
	switch code {
	case 1:
		return "活动未开始"
	case 2:
		return "活动已结束"
	case 3:
		return "库存不足"
	case 4:
		return "不可重复下单"
	default:
		return "下单失败"
	}
}

func toInt64(v interface{}) (int64, error) {
	if v == nil {
		return 0, nil
	}
	var i int64
	_, err := fmt.Sscanf(fmt.Sprintf("%v", v), "%d", &i)
	return i, err
}

var _ = json.Marshal
