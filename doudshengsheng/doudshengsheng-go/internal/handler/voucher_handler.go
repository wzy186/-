package handler

import (
	"context"
	"strconv"

	"doudshengsheng-go/internal/middleware"
	"doudshengsheng-go/internal/service"
	"doudshengsheng-go/internal/utils"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/cloudwego/hertz/pkg/route"
)

type VoucherHandler struct {
	seckillSvc *service.SeckillService
	voucherSvc *service.VoucherService
}

func NewVoucherHandler(seckillSvc *service.SeckillService, voucherSvc *service.VoucherService) *VoucherHandler {
	return &VoucherHandler{seckillSvc: seckillSvc, voucherSvc: voucherSvc}
}

func (h *VoucherHandler) Register(group *route.RouterGroup) {
	group.GET("/voucher/list/:shopId", h.ListByShop)
	group.GET("/voucher/seckill/list/:shopId", h.ListSeckillByShop)
	group.POST("/voucher/order/seckill/:id", h.Seckill)
	group.GET("/voucher/order/my", middleware.LoginRequired(), h.MyOrders)
	// 后台
	group.GET("/admin/voucher/stock/:id", middleware.AdminRequired(), h.Stock)
	group.POST("/admin/voucher/preheat/:id", middleware.AdminRequired(), h.Preheat)
}

func (h *VoucherHandler) ListByShop(c context.Context, ctx *app.RequestContext) {
	shopID, _ := strconv.ParseInt(string(ctx.Param("shopId")), 10, 64)
	ctx.JSON(consts.StatusOK, h.voucherSvc.ListByShop(c, shopID))
}

func (h *VoucherHandler) ListSeckillByShop(c context.Context, ctx *app.RequestContext) {
	shopID, _ := strconv.ParseInt(string(ctx.Param("shopId")), 10, 64)
	ctx.JSON(consts.StatusOK, h.voucherSvc.ListSeckillByShop(c, shopID))
}

func (h *VoucherHandler) Seckill(c context.Context, ctx *app.RequestContext) {
	vid, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	if uid == 0 {
		ctx.JSON(consts.StatusOK, utils.FailWith(401, "未登录"))
		return
	}
	ctx.JSON(consts.StatusOK, h.seckillSvc.Seckill(c, vid, uid))
}

func (h *VoucherHandler) MyOrders(c context.Context, ctx *app.RequestContext) {
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.seckillSvc.MyOrders(c, uid))
}

func (h *VoucherHandler) Stock(c context.Context, ctx *app.RequestContext) {
	vid, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	ctx.JSON(consts.StatusOK, h.voucherSvc.QueryStock(c, vid))
}

func (h *VoucherHandler) Preheat(c context.Context, ctx *app.RequestContext) {
	vid, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	h.voucherSvc.PreheatStock(c, vid)
	ctx.JSON(consts.StatusOK, utils.OK())
}
