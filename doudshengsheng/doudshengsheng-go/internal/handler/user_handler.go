package handler

import (
	"context"
	"fmt"

	"doudshengsheng-go/internal/middleware"
	"doudshengsheng-go/internal/service"
	"doudshengsheng-go/internal/utils"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/cloudwego/hertz/pkg/route"
)

type UserHandler struct {
	svc *service.UserService
}

func NewUserHandler(svc *service.UserService) *UserHandler {
	return &UserHandler{svc: svc}
}

// Register 注册路由
func (h *UserHandler) Register(group *route.RouterGroup) {
	group.POST("/user/code", h.SendCode)
	group.POST("/user/login", h.Login)
	group.GET("/user/me", middleware.LoginRequired(), h.Me)
	group.GET("/user/:id", h.GetByID)
}

func (h *UserHandler) SendCode(c context.Context, ctx *app.RequestContext) {
	phone := string(ctx.Query("phone"))
	ctx.JSON(consts.StatusOK, h.svc.SendCode(c, phone))
}

func (h *UserHandler) Login(c context.Context, ctx *app.RequestContext) {
	var body struct {
		Phone string `json:"phone"`
		Code  string `json:"code"`
	}
	ctx.BindJSON(&body)
	ctx.JSON(consts.StatusOK, h.svc.Login(c, body.Phone, body.Code))
}

func (h *UserHandler) Me(c context.Context, ctx *app.RequestContext) {
	u, _ := ctx.Get("user")
	ctx.JSON(consts.StatusOK, utils.OKWith(u))
}

func (h *UserHandler) GetByID(c context.Context, ctx *app.RequestContext) {
	id := string(ctx.Param("id"))
	var idInt int64
	fmt.Sscanf(id, "%d", &idInt)
	ctx.JSON(consts.StatusOK, h.svc.GetByID(c, idInt))
}
