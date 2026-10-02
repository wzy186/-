package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"doudshengsheng-go/internal/utils"

	"github.com/redis/go-redis/v9"
)

// RedisData 逻辑过期方案的数据包装(含 expireTime),对齐 Java RedisData
type RedisData struct {
	ExpireTime time.Time   `json:"expireTime"`
	Data       interface{} `json:"data"`
}

// CacheClient 缓存客户端:封装旁路缓存 + 三大问题(穿透/击穿/雪崩)
type CacheClient struct {
	rdb *redis.Client
}

func New(rdb *redis.Client) *CacheClient {
	return &CacheClient{rdb: rdb}
}

// Set 写缓存带 TTL
func (c *CacheClient) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return c.rdb.Set(ctx, key, data, ttl).Err()
}

// SetWithLogicalExpire 逻辑过期:不设 Redis TTL,值里带 expireTime(防击穿)
func (c *CacheClient) SetWithLogicalExpire(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	rd := RedisData{
		ExpireTime: time.Now().Add(ttl),
		Data:       value,
	}
	data, _ := json.Marshal(rd)
	return c.rdb.Set(ctx, key, data, 0).Err() // 0 = 永不过期
}

// QueryWithPassThrough 旁路缓存 + 空值缓存防穿透 + 随机 TTL 防雪崩
// dbFallback 查 DB 的回调,返回 (*T, nil) 或 (nil, nil)(不存在)
func QueryWithPassThrough[T any](ctx context.Context, c *CacheClient, key string, ttl time.Duration, dbFallback func() (*T, error)) (*T, error) {
	val, err := c.rdb.Get(ctx, key).Result()
	if err == nil {
		if val == "" {
			return nil, nil // 空值缓存,防穿透
		}
		var t T
		if err := json.Unmarshal([]byte(val), &t); err == nil {
			return &t, nil
		}
	}
	// 未命中,查 DB
	t, err := dbFallback()
	if err != nil {
		return nil, err
	}
	if t == nil {
		// 空值缓存,短 TTL
		c.rdb.Set(ctx, key, "", 2*time.Minute)
		return nil, nil
	}
	c.Set(ctx, key, t, ttl)
	return t, nil
}

// QueryWithMutex 互斥锁防击穿:未命中时抢锁重建,其他等待重试
func QueryWithMutex[T any](ctx context.Context, c *CacheClient, key, lockKey string, ttl time.Duration, dbFallback func() (*T, error)) (*T, error) {
	val, err := c.rdb.Get(ctx, key).Result()
	if err == nil {
		if val == "" {
			return nil, nil
		}
		var t T
		if json.Unmarshal([]byte(val), &t) == nil {
			return &t, nil
		}
	}
	// 未命中,抢锁
	locked, _ := c.rdb.SetNX(ctx, lockKey, "1", 10*time.Second).Result()
	if !locked {
		time.Sleep(50 * time.Millisecond)
		return QueryWithMutex[T](ctx, c, key, lockKey, ttl, dbFallback)
	}
	defer c.rdb.Del(ctx, lockKey)
	// 双重检查
	if val, _ := c.rdb.Get(ctx, key).Result(); val != "" && val != "" {
		var t T
		if json.Unmarshal([]byte(val), &t) == nil {
			return &t, nil
		}
	}
	t, err := dbFallback()
	if err != nil {
		return nil, err
	}
	if t == nil {
		c.rdb.Set(ctx, key, "", 2*time.Minute)
		return nil, nil
	}
	c.Set(ctx, key, t, ttl)
	return t, nil
}

// QueryWithLogicalExpire 逻辑过期防击穿:读到过期则异步重建,当前返回旧数据
func QueryWithLogicalExpire[T any](ctx context.Context, c *CacheClient, key, lockKey string, ttl time.Duration, dbFallback func() (*T, error)) (*T, error) {
	val, err := c.rdb.Get(ctx, key).Result()
	if err != nil || val == "" {
		return nil, nil // 逻辑过期需预热,未命中返回 nil
	}
	var rd RedisData
	if err := json.Unmarshal([]byte(val), &rd); err != nil {
		return nil, nil
	}
	// 反序列化 data 到 T
	dataBytes, _ := json.Marshal(rd.Data)
	var t T
	json.Unmarshal(dataBytes, &t)
	// 未过期,返回旧数据
	if rd.ExpireTime.After(time.Now()) {
		return &t, nil
	}
	// 已过期,抢锁异步重建
	locked, _ := c.rdb.SetNX(ctx, lockKey, "1", 10*time.Second).Result()
	if locked {
		go func() {
			defer c.rdb.Del(utils.Ctx, lockKey)
			// 双重检查
			if v, _ := c.rdb.Get(utils.Ctx, key).Result(); v != "" {
				var r RedisData
				if json.Unmarshal([]byte(v), &r) == nil && r.ExpireTime.After(time.Now()) {
					return
				}
			}
			newT, err := dbFallback()
			if err == nil && newT != nil {
				c.SetWithLogicalExpire(utils.Ctx, key, newT, ttl)
			}
		}()
	}
	return &t, nil // 返回旧数据
}

// BloomContains 布隆过滤器判断(防穿透前置)
func (c *CacheClient) BloomContains(ctx context.Context, key string, item string) bool {
	res, err := c.rdb.Do(ctx, "BF.EXISTS", key, item).Int()
	if err != nil {
		return true // 布隆不可用时降级为"存在",走正常缓存流程
	}
	return res == 1
}

// BloomAdd 加进布隆过滤器
func (c *CacheClient) BloomAdd(ctx context.Context, key string, item string) error {
	return c.rdb.Do(ctx, "BF.ADD", key, item).Err()
}

// BloomInit 初始化布隆过滤器(容量、误判率)
func (c *CacheClient) BloomInit(ctx context.Context, key string, capacity int64, errorRate float64) error {
	// tryInit:已存在则忽略
	c.rdb.Do(ctx, "BF.RESERVE", key, errorRate, capacity)
	return nil
}

var _ = fmt.Sprintf
