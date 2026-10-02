package middleware

import (
	"context"
	"fmt"
	"strings"
	"time"

	"doudshengsheng-go/internal/utils"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/common/hlog"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
)

var _ = strings.TrimSpace
var _ = context.Background

// 常量:返回 JSON
func writeJSON(c context.Context, ctx *app.RequestContext, code int, obj interface{}) {
	ctx.JSON(code, obj)
}

// RefreshToken 刷新 token + 把用户放进 context(对应 Java RefreshTokenInterceptor)
// 拦截所有请求,只读取续期,不拦截
func RefreshToken() app.HandlerFunc {
	return func(c context.Context, ctx *app.RequestContext) {
		token := string(ctx.GetHeader("authorization"))
		if token == "" {
			ctx.Next(c)
			return
		}
		// 从 Redis Hash 取用户
		fields, err := utils.Redis.HGetAll(c, utils.LoginTokenKey+token).Result()
		if err != nil || len(fields) == 0 {
			ctx.Next(c)
			return
		}
		u := &utils.UserDTO{
			NickName: fields["nickName"],
			Icon:     fields["icon"],
		}
		fmt.Sscanf(fields["id"], "%d", &u.ID)
		if r, ok := fields["role"]; ok {
			fmt.Sscanf(r, "%d", &u.Role)
		}
		// 存进 context(对应 ThreadLocal)
		c = utils.SaveUser(c, u)
		// 刷新 token 有效期
		utils.Redis.Expire(c, utils.LoginTokenKey+token, utils.LoginTokenTTL*time.Second)
		ctx.Set("user", u)
		ctx.Next(c)
	}
}

// LoginRequired 校验登录(对应 Java LoginInterceptor)
func LoginRequired() app.HandlerFunc {
	return func(c context.Context, ctx *app.RequestContext) {
		u, _ := ctx.Get("user")
		if u == nil {
			writeJSON(c, ctx, consts.StatusUnauthorized, utils.FailWith(401, "未登录"))
			ctx.Abort()
			return
		}
		ctx.Next(c)
	}
}

// AdminRequired 校验管理员(对应 Java @AdminOnly 切面)
func AdminRequired() app.HandlerFunc {
	return func(c context.Context, ctx *app.RequestContext) {
		u, _ := ctx.Get("user")
		ud, ok := u.(*utils.UserDTO)
		if !ok || ud == nil {
			writeJSON(c, ctx, consts.StatusUnauthorized, utils.FailWith(401, "未登录"))
			ctx.Abort()
			return
		}
		if ud.Role != 1 {
			writeJSON(c, ctx, consts.StatusForbidden, utils.FailWith(403, "无权限,仅管理员可访问"))
			ctx.Abort()
			return
		}
		ctx.Next(c)
	}
}

// Recover 全局异常恢复(对应 Java GlobalExceptionHandler 兜底)
func Recover() app.HandlerFunc {
	return func(c context.Context, ctx *app.RequestContext) {
		defer func() {
			if r := recover(); r != nil {
				hlog.Error("未捕获异常: ", r)
				writeJSON(c, ctx, consts.StatusInternalServerError, utils.Fail("服务异常,请稍后重试"))
				ctx.Abort()
			}
		}()
		ctx.Next(c)
	}
}

// CORS 跨域(前端 Vue 在 5173,Go 在 8082)
func CORS() app.HandlerFunc {
	return func(c context.Context, ctx *app.RequestContext) {
		ctx.Header("Access-Control-Allow-Origin", "*")
		ctx.Header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
		ctx.Header("Access-Control-Allow-Headers", "Content-Type,Authorization")
		if string(ctx.Method()) == "OPTIONS" {
			ctx.SetStatusCode(consts.StatusNoContent)
			ctx.Abort()
			return
		}
		// 允许自定义 header authorization 透传
		_ = strings.TrimSpace
		ctx.Next(c)
	}
}
