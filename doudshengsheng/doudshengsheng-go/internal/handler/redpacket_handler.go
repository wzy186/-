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

type RedPacketHandler struct {
	svc *service.RedPacketService
}

func NewRedPacketHandler(svc *service.RedPacketService) *RedPacketHandler {
	return &RedPacketHandler{svc: svc}
}

func (h *RedPacketHandler) Register(group *route.RouterGroup) {
	group.POST("/redpacket/create", middleware.LoginRequired(), h.Create)
	group.POST("/redpacket/grab/:id", h.Grab)
	group.GET("/redpacket/rank/:id", h.Rank)
	group.GET("/admin/redpacket/list", middleware.AdminRequired(), h.List)
	group.GET("/admin/redpacket/:id", middleware.AdminRequired(), h.Detail)
}

func (h *RedPacketHandler) Create(c context.Context, ctx *app.RequestContext) {
	title := string(ctx.Query("title"))
	totalYuan, _ := strconv.Atoi(string(ctx.Query("totalYuan")))
	count, _ := strconv.Atoi(string(ctx.Query("count")))
	ctx.JSON(consts.StatusOK, h.svc.Create(c, title, totalYuan, count))
}

func (h *RedPacketHandler) Grab(c context.Context, ctx *app.RequestContext) {
	rpID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	if uid == 0 {
		ctx.JSON(consts.StatusOK, utils.FailWith(401, "未登录"))
		return
	}
	ctx.JSON(consts.StatusOK, h.svc.Grab(c, rpID, uid))
}

func (h *RedPacketHandler) Rank(c context.Context, ctx *app.RequestContext) {
	rpID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	ctx.JSON(consts.StatusOK, h.svc.Rank(c, rpID))
}

func (h *RedPacketHandler) List(c context.Context, ctx *app.RequestContext) {
	ctx.JSON(consts.StatusOK, h.svc.ListRedPackets(c))
}

func (h *RedPacketHandler) Detail(c context.Context, ctx *app.RequestContext) {
	rpID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	ctx.JSON(consts.StatusOK, h.svc.RedPacketDetail(c, rpID))
}
