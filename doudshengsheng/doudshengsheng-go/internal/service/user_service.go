package service

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"

	"gorm.io/gorm"
)

type UserService struct{}

// SendCode 发送验证码:存 Redis,2分钟过期
func (s *UserService) SendCode(ctx context.Context, phone string) *utils.Result {
	if !isPhoneValid(phone) {
		return utils.FailWith(400, "手机号格式错误")
	}
	code := fmt.Sprintf("%06d", rand.Intn(1000000))
	utils.Redis.Set(ctx, utils.LoginCodeKey+phone, code, utils.LoginCodeTTL*time.Second)
	fmt.Println("【兜省省-Go】验证码:", code)
	return utils.OK()
}

// Login 登录:校验验证码 → 查/建用户 → 生成 token 存 Redis
func (s *UserService) Login(ctx context.Context, phone, code string) *utils.Result {
	if !isPhoneValid(phone) {
		return utils.FailWith(400, "手机号格式错误")
	}
	cacheCode, err := utils.Redis.Get(ctx, utils.LoginCodeKey+phone).Result()
	if err != nil || cacheCode != code {
		return utils.Fail("验证码错误或已过期")
	}
	// 查用户,不存在则注册
	var user model.User
	err = utils.DB.Where("phone = ?", phone).First(&user).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		user = model.User{Phone: phone, NickName: "兜省省用户_" + randStr(6)}
		utils.DB.Create(&user)
	} else if err != nil {
		return utils.Fail("查询用户失败")
	}
	// 生成 token,存 Redis Hash
	token := randStr(32)
	fields := map[string]interface{}{
		"id":       fmt.Sprintf("%d", user.ID),
		"nickName": user.NickName,
		"icon":     user.Icon,
		"role":     fmt.Sprintf("%d", user.Role),
	}
	utils.Redis.HSet(ctx, utils.LoginTokenKey+token, fields)
	utils.Redis.Expire(ctx, utils.LoginTokenKey+token, utils.LoginTokenTTL*time.Second)
	return utils.OKWith(token)
}

// GetByID 查用户
func (s *UserService) GetByID(ctx context.Context, id int64) *utils.Result {
	var user model.User
	if err := utils.DB.First(&user, id).Error; err != nil {
		return utils.Fail("用户不存在")
	}
	user.Password = ""
	return utils.OKWith(user)
}

func isPhoneValid(phone string) bool {
	if len(phone) != 11 || phone[0] != '1' {
		return false
	}
	for _, c := range phone {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func randStr(n int) string {
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}
