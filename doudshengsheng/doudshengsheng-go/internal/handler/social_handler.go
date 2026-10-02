package handler

import (
	"context"
	"strconv"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/service"
	"doudshengsheng-go/internal/utils"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/cloudwego/hertz/pkg/route"
)

type SocialHandler struct {
	blogSvc   *service.BlogService
	followSvc *service.FollowService
	statsSvc  *service.StatsService
}

func NewSocialHandler(blogSvc *service.BlogService, followSvc *service.FollowService, statsSvc *service.StatsService) *SocialHandler {
	return &SocialHandler{blogSvc: blogSvc, followSvc: followSvc, statsSvc: statsSvc}
}

func (h *SocialHandler) Register(group *route.RouterGroup) {
	// 笔记
	group.GET("/blog/hot", h.Hot)
	group.GET("/blog/:id", h.BlogDetail)
	group.PUT("/blog/like/:id", h.Like)
	group.GET("/blog/likes/:id", h.Likes)
	group.POST("/blog", h.SaveBlog)
	group.GET("/blog/of/user/:userId", h.BlogOfUser)
	// 关注
	group.PUT("/follow/:id/:isFollow", h.Follow)
	group.GET("/follow/or/:id", h.IsFollow)
	group.GET("/follow/common/:id", h.CommonFollows)
	group.GET("/follow/feed", h.Feed)
	group.GET("/follow/profile/:id", h.Profile)
	// 签到/UV
	group.POST("/stats/sign", h.Sign)
	group.GET("/stats/sign/count", h.SignCount)
	group.GET("/stats/sign/records", h.SignRecords)
	group.POST("/stats/uv", h.UV)
	group.GET("/stats/uv/count", h.UVCount)
}

// --- 笔记 ---
func (h *SocialHandler) Hot(c context.Context, ctx *app.RequestContext) {
	current, _ := strconv.Atoi(string(ctx.DefaultQuery("current", "1")))
	ctx.JSON(consts.StatusOK, h.blogSvc.Hot(c, current))
}

func (h *SocialHandler) BlogDetail(c context.Context, ctx *app.RequestContext) {
	id, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.blogSvc.Detail(c, id, uid))
}

func (h *SocialHandler) Like(c context.Context, ctx *app.RequestContext) {
	id, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.blogSvc.Like(c, id, uid))
}

func (h *SocialHandler) Likes(c context.Context, ctx *app.RequestContext) {
	id, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	ctx.JSON(consts.StatusOK, h.blogSvc.Likes(c, id))
}

func (h *SocialHandler) SaveBlog(c context.Context, ctx *app.RequestContext) {
	var blog model.Blog
	ctx.BindJSON(&blog)
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.blogSvc.Save(c, &blog, uid))
}

func (h *SocialHandler) BlogOfUser(c context.Context, ctx *app.RequestContext) {
	uid, _ := strconv.ParseInt(string(ctx.Param("userId")), 10, 64)
	ctx.JSON(consts.StatusOK, h.blogSvc.OfUser(c, uid))
}

// --- 关注 ---
func (h *SocialHandler) Follow(c context.Context, ctx *app.RequestContext) {
	targetID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	isFollow := string(ctx.Param("isFollow")) == "true"
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.followSvc.Follow(c, uid, targetID, isFollow))
}

func (h *SocialHandler) IsFollow(c context.Context, ctx *app.RequestContext) {
	targetID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.followSvc.IsFollow(c, uid, targetID))
}

func (h *SocialHandler) CommonFollows(c context.Context, ctx *app.RequestContext) {
	targetID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.followSvc.CommonFollows(c, uid, targetID))
}

func (h *SocialHandler) Feed(c context.Context, ctx *app.RequestContext) {
	max, _ := strconv.ParseInt(string(ctx.Query("max")), 10, 64)
	offset, _ := strconv.Atoi(string(ctx.Query("offset")))
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.followSvc.Feed(c, uid, max, offset))
}

func (h *SocialHandler) Profile(c context.Context, ctx *app.RequestContext) {
	targetID, _ := strconv.ParseInt(string(ctx.Param("id")), 10, 64)
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.followSvc.UserProfile(c, targetID, uid))
}

// --- 签到/UV ---
func (h *SocialHandler) Sign(c context.Context, ctx *app.RequestContext) {
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.statsSvc.Sign(c, uid))
}

func (h *SocialHandler) SignCount(c context.Context, ctx *app.RequestContext) {
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.statsSvc.SignCount(c, uid))
}

func (h *SocialHandler) SignRecords(c context.Context, ctx *app.RequestContext) {
	uid := utils.GetUserID(c)
	ctx.JSON(consts.StatusOK, h.statsSvc.SignRecords(c, uid))
}

func (h *SocialHandler) UV(c context.Context, ctx *app.RequestContext) {
	bizKey := string(ctx.Query("bizKey"))
	uid, _ := strconv.ParseInt(string(ctx.Query("userId")), 10, 64)
	ctx.JSON(consts.StatusOK, h.statsSvc.UV(c, bizKey, uid))
}

func (h *SocialHandler) UVCount(c context.Context, ctx *app.RequestContext) {
	bizKey := string(ctx.Query("bizKey"))
	ctx.JSON(consts.StatusOK, h.statsSvc.UVCount(c, bizKey))
}
