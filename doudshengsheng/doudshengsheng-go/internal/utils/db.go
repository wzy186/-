package utils

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var (
	DB    *gorm.DB
	Redis *redis.Client
	Ctx   = context.Background()
)

// InitDB 初始化 MySQL(GORM),复用 Java 版同一套表
func InitDB(dsn string) error {
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Warn), // 只打 warning 以上
	})
	if err != nil {
		return fmt.Errorf("连接 MySQL 失败: %w", err)
	}
	DB = db
	return nil
}

// InitRedis 初始化 Redis
func InitRedis(addr, password string, db int) error {
	Redis = redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
		PoolSize: 20,
	})
	if err := Redis.Ping(Ctx).Err(); err != nil {
		return fmt.Errorf("连接 Redis 失败: %w", err)
	}
	return nil
}
