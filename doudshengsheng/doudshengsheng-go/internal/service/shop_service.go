package service

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"time"

	"doudshengsheng-go/internal/cache"
	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"

	"github.com/redis/go-redis/v9"
)

type ShopService struct {
	cc *cache.CacheClient
}

func NewShopService(cc *cache.CacheClient) *ShopService {
	return &ShopService{cc: cc}
}

// PreheatBloom 启动预热:把所有商铺 id 加进布隆过滤器
func (s *ShopService) PreheatBloom(ctx context.Context) error {
	s.cc.BloomInit(ctx, utils.BloomShopKey, 100000, 0.01)
	var shops []model.Shop
	utils.DB.Find(&shops)
	for _, shop := range shops {
		s.cc.BloomAdd(ctx, utils.BloomShopKey, strconv.FormatInt(shop.ID, 10))
	}
	// GEO 预热
	for _, shop := range shops {
		if shop.X != 0 && shop.Y != 0 {
			utils.Redis.GeoAdd(ctx, utils.ShopGEOKey, &redis.GeoLocation{
				Name:      strconv.FormatInt(shop.ID, 10),
				Longitude: shop.X,
				Latitude:  shop.Y,
			})
		}
	}
	fmt.Printf("布隆+GEO 预热完成,商铺 %d 家\n", len(shops))
	return nil
}

// QueryByID 查商铺:strategy 切换三策略
func (s *ShopService) QueryByID(ctx context.Context, id int64, strategy string) *utils.Result {
	// 布隆前置:不存在直接返回,不查缓存不查 DB
	if !s.cc.BloomContains(ctx, utils.BloomShopKey, strconv.FormatInt(id, 10)) {
		return utils.Fail("商铺不存在")
	}

	key := utils.CacheShopKey + strconv.FormatInt(id, 10)
	lockKey := utils.LockShopKey + strconv.FormatInt(id, 10)
	ttl := time.Duration(utils.CacheShopTTL) * time.Minute

	dbFallback := func() (*model.Shop, error) {
		var shop model.Shop
		err := utils.DB.First(&shop, id).Error
		if err != nil {
			return nil, nil
		}
		return &shop, nil
	}

	var shop *model.Shop
	var err error
	switch strategy {
	case "mutex":
		shop, err = cache.QueryWithMutex[model.Shop](ctx, s.cc, key, lockKey, ttl, dbFallback)
	case "logical":
		logicKey := utils.CacheShopLogicKey + strconv.FormatInt(id, 10)
		shop, err = cache.QueryWithLogicalExpire[model.Shop](ctx, s.cc, logicKey, lockKey, ttl, dbFallback)
		if shop == nil {
			// 逻辑过期需预热:首次未命中主动写入
			dbShop, _ := dbFallback()
			if dbShop != nil {
				s.cc.SetWithLogicalExpire(ctx, logicKey, dbShop, ttl)
				shop = dbShop
			}
		}
	default:
		// 旁路 + 空值防穿透 + 随机 TTL 防雪崩
		randomTTL := ttl + time.Duration(rand.Intn(10))*time.Minute
		shop, err = cache.QueryWithPassThrough[model.Shop](ctx, s.cc, key, randomTTL, dbFallback)
	}
	if err != nil {
		return utils.Fail("查询失败")
	}
	if shop == nil {
		return utils.Fail("商铺不存在")
	}
	return utils.OKWith(shop)
}

// Update 更新商铺:先更 DB 再删缓存
func (s *ShopService) Update(ctx context.Context, shop *model.Shop) *utils.Result {
	if shop.ID == 0 {
		return utils.Fail("商铺 id 不能为空")
	}
	utils.DB.Model(shop).Updates(shop)
	utils.Redis.Del(ctx, utils.CacheShopKey+strconv.FormatInt(shop.ID, 10))
	utils.Redis.Del(ctx, utils.CacheShopLogicKey+strconv.FormatInt(shop.ID, 10))
	return utils.OK()
}

// Save 新增商铺:写 DB + 加布隆 + 加 GEO
func (s *ShopService) Save(ctx context.Context, shop *model.Shop) *utils.Result {
	if shop.Name == "" || shop.TypeID == 0 {
		return utils.Fail("商铺名和类型不能为空")
	}
	utils.DB.Create(shop)
	s.cc.BloomAdd(ctx, utils.BloomShopKey, strconv.FormatInt(shop.ID, 10))
	if shop.X != 0 && shop.Y != 0 {
		utils.Redis.GeoAdd(ctx, utils.ShopGEOKey, &redis.GeoLocation{
			Name: strconv.FormatInt(shop.ID, 10), Longitude: shop.X, Latitude: shop.Y,
		})
	}
	return utils.OKWith(shop.ID)
}

// QueryByType 按类型查
func (s *ShopService) QueryByType(ctx context.Context, typeID int64) *utils.Result {
	var shops []model.Shop
	utils.DB.Where("type_id = ?", typeID).Find(&shops)
	return utils.OKWith(shops)
}

// QueryTypeList 商铺类型列表(列表型缓存)
func (s *ShopService) QueryTypeList(ctx context.Context) *utils.Result {
	key := utils.CacheShopKey + "type:list"
	val, err := utils.Redis.Get(ctx, key).Result()
	if err == nil && val != "" {
		var types []model.ShopType
		utils.UnmarshalJSON([]byte(val), &types)
		return utils.OKWith(types)
	}
	var types []model.ShopType
	utils.DB.Order("sort").Find(&types)
	utils.Redis.Set(ctx, key, utils.ToJSON(types), 30*time.Minute)
	return utils.OKWith(types)
}

// QueryNearby 附近商铺(GEO)
func (s *ShopService) QueryNearby(ctx context.Context, typeID int64, x, y, distKm float64) *utils.Result {
	res, err := utils.Redis.GeoSearchLocation(ctx, utils.ShopGEOKey, &redis.GeoSearchLocationQuery{
		GeoSearchQuery: redis.GeoSearchQuery{
			Longitude:  x,
			Latitude:   y,
			Radius:     distKm,
			RadiusUnit: "km",
			Sort:       "ASC",
			Count:      10,
		},
		WithDist: true,
	}).Result()
	if err != nil || len(res) == 0 {
		return utils.OKWith([]interface{}{})
	}
	ids := make([]int64, 0, len(res))
	distMap := map[int64]float64{}
	for _, loc := range res {
		id, _ := strconv.ParseInt(loc.Name, 10, 64)
		ids = append(ids, id)
		distMap[id] = loc.Dist
	}
	var shops []model.Shop
	utils.DB.Where("id IN ? AND type_id = ?", ids, typeID).Find(&shops)
	return utils.OKWith(map[string]interface{}{
		"shops":    shops,
		"distances": distMap,
	})
}
