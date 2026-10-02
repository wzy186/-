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

type FollowService struct{}

// Follow 关注/取关(DB + Redis Set 双写)
func (s *FollowService) Follow(ctx context.Context, userID, targetID int64, isFollow bool) *utils.Result {
	key := utils.FollowKey + strconv.FormatInt(userID, 10)
	if isFollow {
		f := model.Follow{UserID: userID, FollowUserID: targetID}
		if err := utils.DB.Create(&f).Error; err == nil {
			utils.Redis.SAdd(ctx, key, strconv.FormatInt(targetID, 10))
		}
	} else {
		utils.DB.Where("user_id = ? AND follow_user_id = ?", userID, targetID).Delete(&model.Follow{})
		utils.Redis.SRem(ctx, key, strconv.FormatInt(targetID, 10))
	}
	return utils.OK()
}

// IsFollow 是否已关注(优先 Redis,降级 DB)
func (s *FollowService) IsFollow(ctx context.Context, userID, targetID int64) *utils.Result {
	isMember, err := utils.Redis.SIsMember(ctx, utils.FollowKey+strconv.FormatInt(userID, 10), strconv.FormatInt(targetID, 10)).Result()
	if err == nil && isMember {
		return utils.OKWith(true)
	}
	var cnt int64
	utils.DB.Model(&model.Follow{}).Where("user_id = ? AND follow_user_id = ?", userID, targetID).Count(&cnt)
	return utils.OKWith(cnt > 0)
}

// CommonFollows 共同关注(Redis Set 交集,降级 DB)
func (s *FollowService) CommonFollows(ctx context.Context, userID, targetID int64) *utils.Result {
	intersect, err := utils.Redis.SInter(ctx,
		utils.FollowKey+strconv.FormatInt(userID, 10),
		utils.FollowKey+strconv.FormatInt(targetID, 10),
	).Result()
	if err == nil && len(intersect) > 0 {
		ids := []int64{}
		for _, v := range intersect {
			id, _ := strconv.ParseInt(v, 10, 64)
			ids = append(ids, id)
		}
		return utils.OKWith(ids)
	}
	// 降级 DB
	var myFollows, targetFollows []int64
	utils.DB.Model(&model.Follow{}).Where("user_id = ?", userID).Pluck("follow_user_id", &myFollows)
	utils.DB.Model(&model.Follow{}).Where("user_id = ?", targetID).Pluck("follow_user_id", &targetFollows)
	mySet := map[int64]bool{}
	for _, id := range myFollows {
		mySet[id] = true
	}
	common := []int64{}
	for _, id := range targetFollows {
		if mySet[id] {
			common = append(common, id)
		}
	}
	return utils.OKWith(common)
}

// PushToFollowers Feed 推模式:发笔记时推送到粉丝收件箱 ZSet
func (s *FollowService) PushToFollowers(ctx context.Context, blogID, authorID int64) {
	var fans []int64
	utils.DB.Model(&model.Follow{}).Where("follow_user_id = ?", authorID).Pluck("user_id", &fans)
	now := float64(time.Now().UnixMilli())
	for _, fan := range fans {
		utils.Redis.ZAdd(ctx, utils.FeedKey+strconv.FormatInt(fan, 10), redis.Z{
			Score:  now, Member: strconv.FormatInt(blogID, 10),
		})
	}
}

// Feed 收件箱滚动分页(推模式)
func (s *FollowService) Feed(ctx context.Context, userID int64, max int64, offset int) *utils.Result {
	key := utils.FeedKey + strconv.FormatInt(userID, 10)
	if max == 0 {
		max = time.Now().UnixMilli()
	}
	if offset == 0 {
		offset = 0
	}
	res, err := utils.Redis.ZRevRangeByScoreWithScores(ctx, key, &redis.ZRangeBy{
		Max:   strconv.FormatInt(max, 10),
		Min:   "0",
		Offset: int64(offset), Count: 5,
	}).Result()
	if err != nil || len(res) == 0 {
		return utils.OKWith(map[string]interface{}{"list": []int64{}, "minTime": 0, "offset": 0})
	}
	list := []int64{}
	minTime := int64(0)
	os := 0
	for _, z := range res {
		id, _ := strconv.ParseInt(fmt.Sprintf("%v", z.Member), 10, 64)
		list = append(list, id)
		t := int64(z.Score)
		if t == minTime {
			os++
		} else if t < minTime || minTime == 0 {
			minTime = t
			os = 1
		}
	}
	return utils.OKWith(map[string]interface{}{"list": list, "minTime": minTime, "offset": os})
}

// UserProfile 用户主页:关注数/粉丝数/是否已关注
func (s *FollowService) UserProfile(ctx context.Context, userID, currentUID int64) *utils.Result {
	var following, followers int64
	utils.DB.Model(&model.Follow{}).Where("user_id = ?", userID).Count(&following)
	utils.DB.Model(&model.Follow{}).Where("follow_user_id = ?", userID).Count(&followers)
	isFollowing := false
	isMe := currentUID == userID
	if currentUID != 0 && !isMe {
		var cnt int64
		utils.DB.Model(&model.Follow{}).Where("user_id = ? AND follow_user_id = ?", currentUID, userID).Count(&cnt)
		isFollowing = cnt > 0
	}
	return utils.OKWith(map[string]interface{}{
		"userId": userID, "following": following, "followers": followers,
		"isFollowing": isFollowing, "isMe": isMe,
	})
}
