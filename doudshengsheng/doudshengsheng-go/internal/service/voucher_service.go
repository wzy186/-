package service

import (
	"context"
	"fmt"
	"strconv"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"
)

type VoucherService struct{}

// PreheatStock 预热秒杀券库存到 Redis
func (s *VoucherService) PreheatStock(ctx context.Context, voucherID int64) {
	var sv model.SeckillVoucher
	if err := utils.DB.First(&sv, voucherID).Error; err != nil {
		return
	}
	utils.Redis.Set(ctx, utils.SeckillStockKey+strconv.FormatInt(voucherID, 10), sv.Stock, 0)
	fmt.Printf("秒杀券 %d 库存预热: %d\n", voucherID, sv.Stock)
}

// PreheatAll 预热所有秒杀券
func (s *VoucherService) PreheatAll(ctx context.Context) {
	var svs []model.SeckillVoucher
	utils.DB.Find(&svs)
	for _, sv := range svs {
		utils.Redis.Set(ctx, utils.SeckillStockKey+strconv.FormatInt(sv.VoucherID, 10), sv.Stock, 0)
	}
	fmt.Printf("秒杀券库存预热完成,共 %d 张\n", len(svs))
}

// QueryStock 查 Redis/DB 库存对比
func (s *VoucherService) QueryStock(ctx context.Context, voucherID int64) *utils.Result {
	redisStock, _ := utils.Redis.Get(ctx, utils.SeckillStockKey+strconv.FormatInt(voucherID, 10)).Int()
	var sv model.SeckillVoucher
	dbStock := 0
	if err := utils.DB.First(&sv, voucherID).Error; err == nil {
		dbStock = sv.Stock
	}
	return utils.OKWith(map[string]int{
		"redisStock": redisStock,
		"dbStock":    dbStock,
	})
}

// ListByShop 按商铺查券
func (s *VoucherService) ListByShop(ctx context.Context, shopID int64) *utils.Result {
	var vouchers []model.Voucher
	utils.DB.Where("shop_id = ?", shopID).Find(&vouchers)
	return utils.OKWith(vouchers)
}

// ListSeckillByShop 按商铺查秒杀券
func (s *VoucherService) ListSeckillByShop(ctx context.Context, shopID int64) *utils.Result {
	var vouchers []model.Voucher
	utils.DB.Where("shop_id = ? AND type = 2", shopID).Find(&vouchers)
	return utils.OKWith(vouchers)
}
