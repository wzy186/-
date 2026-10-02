package service

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"

	"github.com/redis/go-redis/v9"
)

type BlogService struct {
	followSvc *FollowService
}

func NewBlogService(followSvc *FollowService) *BlogService {
	return &BlogService{followSvc: followSvc}
}

// Hot 热门笔记(带作者名)
func (s *BlogService) Hot(ctx context.Context, current int) *utils.Result {
	if current == 0 {
		current = 1
	}
	var blogs []model.Blog
	utils.DB.Order("liked desc").Offset((current - 1) * 5).Limit(5).Find(&blogs)
	// 带作者名
	list := []map[string]interface{}{}
	for _, b := range blogs {
		var user model.User
		utils.DB.First(&user, b.UserID)
		list = append(list, map[string]interface{}{
			"id":         b.ID,
			"title":      b.Title,
			"content":    b.Content,
			"shopId":     b.ShopID,
			"liked":      b.Liked,
			"userId":     b.UserID,
			"authorName": user.NickName,
		})
	}
	return utils.OKWith(list)
}

// Detail 笔记详情(含作者 + 是否已赞)
func (s *BlogService) Detail(ctx context.Context, id, currentUID int64) *utils.Result {
	var blog model.Blog
	if err := utils.DB.First(&blog, id).Error; err != nil {
		return utils.Fail("笔记不存在")
	}
	var author model.User
	utils.DB.First(&author, blog.UserID)
	isLike := false
	if currentUID != 0 {
		score, err := utils.Redis.ZScore(ctx, utils.BlogLikedKey+strconv.FormatInt(id, 10), strconv.FormatInt(currentUID, 10)).Result()
		isLike = err == nil && score != 0
	}
	return utils.OKWith(map[string]interface{}{
		"id": blog.ID, "title": blog.Title, "content": blog.Content,
		"shopId": blog.ShopID, "liked": blog.Liked, "userId": blog.UserID,
		"author": map[string]interface{}{"id": author.ID, "nickName": author.NickName, "icon": author.Icon},
		"isLike": isLike,
	})
}

// Like 点赞/取消(ZSet 存 uid+时间戳,一人一赞)
func (s *BlogService) Like(ctx context.Context, id, userID int64) *utils.Result {
	key := utils.BlogLikedKey + strconv.FormatInt(id, 10)
	uidStr := strconv.FormatInt(userID, 10)
	score, err := utils.Redis.ZScore(ctx, key, uidStr).Result()
	if err == nil && score != 0 {
		// 已赞 → 取消
		utils.DB.Model(&model.Blog{}).Where("id = ?", id).UpdateColumn("liked", "liked - 1")
		utils.Redis.ZRem(ctx, key, uidStr)
	} else {
		// 未赞 → 点赞
		utils.DB.Model(&model.Blog{}).Where("id = ?", id).UpdateColumn("liked", "liked + 1")
		utils.Redis.ZAdd(ctx, key, redis.Z{Score: float64(time.Now().UnixMilli()), Member: uidStr})
	}
	return utils.OK()
}

// Likes 点赞排行榜 Top5(按时间)
func (s *BlogService) Likes(ctx context.Context, id int64) *utils.Result {
	key := utils.BlogLikedKey + strconv.FormatInt(id, 10)
	members, _ := utils.Redis.ZRangeWithScores(ctx, key, 0, 4).Result()
	list := []map[string]interface{}{}
	for _, m := range members {
		uid, _ := strconv.ParseInt(fmt.Sprintf("%v", m.Member), 10, 64)
		list = append(list, map[string]interface{}{
			"userId": uid, "likeTime": int64(m.Score),
		})
	}
	return utils.OKWith(list)
}

// Save 发布笔记 + Feed 推送给粉丝
func (s *BlogService) Save(ctx context.Context, blog *model.Blog, userID int64) *utils.Result {
	blog.UserID = userID
	blog.Liked = 0
	utils.DB.Create(blog)
	// Feed 推模式:推送给粉丝
	s.followSvc.PushToFollowers(ctx, blog.ID, userID)
	return utils.OKWith(blog.ID)
}

// OfUser 按用户查笔记(个人主页)
func (s *BlogService) OfUser(ctx context.Context, userID int64) *utils.Result {
	var blogs []model.Blog
	utils.DB.Where("user_id = ?", userID).Order("create_time desc").Find(&blogs)
	return utils.OKWith(blogs)
}
