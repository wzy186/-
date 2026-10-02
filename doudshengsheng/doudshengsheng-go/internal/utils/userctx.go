package utils

import (
	"context"
)

// UserDTO 脱敏用户信息,放在 context 里传递(对应 Java 的 ThreadLocal + UserHolder)
type UserDTO struct {
	ID       int64  `json:"id"`
	NickName string `json:"nickName"`
	Icon     string `json:"icon"`
	Role     int    `json:"role"` // 0普通 1管理员
}

type ctxKey struct{}

// SaveUser 把用户存进 context
func SaveUser(ctx context.Context, u *UserDTO) context.Context {
	return context.WithValue(ctx, ctxKey{}, u)
}

// GetUser 从 context 取用户
func GetUser(ctx context.Context) *UserDTO {
	u, _ := ctx.Value(ctxKey{}).(*UserDTO)
	return u
}

// GetUserID 取用户ID
func GetUserID(ctx context.Context) int64 {
	u := GetUser(ctx)
	if u == nil {
		return 0
	}
	return u.ID
}
