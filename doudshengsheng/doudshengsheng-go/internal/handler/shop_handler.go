package handler

import (
	"context"
	"fmt"
	"strconv"

	"doudshengsheng-go/internal/middleware"
	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/service"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/cloudwego/hertz/pkg/route"
)

type ShopHandler struct {
	svc *service.ShopService
}

func NewShopHandler(svc *service.ShopService) *ShopHandler {
	return &ShopHandler{svc: svc}
}

func (h *ShopHandler) Register(group *route.RouterGroup) {
	group.GET("/shop/:id", h.QueryByID)
	group.PUT("/shop", middleware.AdminRequired(), h.Update)
	group.POST("/shop", middleware.AdminRequired(), h.Save)
	group.GET("/shop/of/type", h.QueryByType)
	group.GET("/shop/of/near", h.QueryNearby)
	group.GET("/shop/type/list", h.QueryTypeList)
}

func (h *ShopHandler) QueryByID(c context.Context, ctx *app.RequestContext) {
	idStr := string(ctx.Param("id"))
	id, _ := strconv.ParseInt(idStr, 10, 64)
	strategy := string(ctx.DefaultQuery("strategy", "pass-through"))
	ctx.JSON(consts.StatusOK, h.svc.QueryByID(c, id, strategy))
}

func (h *ShopHandler) Update(c context.Context, ctx *app.RequestContext) {
	var shop model.Shop
	ctx.BindJSON(&shop)
	ctx.JSON(consts.StatusOK, h.svc.Update(c, &shop))
}

func (h *ShopHandler) Save(c context.Context, ctx *app.RequestContext) {
	var shop model.Shop
	ctx.BindJSON(&shop)
	ctx.JSON(consts.StatusOK, h.svc.Save(c, &shop))
}

func (h *ShopHandler) QueryByType(c context.Context, ctx *app.RequestContext) {
	typeID, _ := strconv.ParseInt(string(ctx.Query("typeId")), 10, 64)
	ctx.JSON(consts.StatusOK, h.svc.QueryByType(c, typeID))
}

func (h *ShopHandler) QueryTypeList(c context.Context, ctx *app.RequestContext) {
	ctx.JSON(consts.StatusOK, h.svc.QueryTypeList(c))
}

func (h *ShopHandler) QueryNearby(c context.Context, ctx *app.RequestContext) {
	typeID, _ := strconv.ParseInt(string(ctx.Query("typeId")), 10, 64)
	x, _ := strconv.ParseFloat(string(ctx.Query("x")), 64)
	y, _ := strconv.ParseFloat(string(ctx.Query("y")), 64)
	distKm, _ := strconv.ParseFloat(string(ctx.DefaultQuery("distKm", "5")), 64)
	fmt.Println(typeID, x, y, distKm)
	ctx.JSON(consts.StatusOK, h.svc.QueryNearby(c, typeID, x, y, distKm))
}
