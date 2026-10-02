package utils

// Redis key 前缀,对齐 Java 版 RedisConstants
const (
	Prefix = "dss-go:" // Go 版用 dss-go 前缀,和 Java 版 dss: 隔离

	LoginCodeKey     = Prefix + "login:code:"
	LoginCodeTTL     = 120
	LoginTokenKey    = Prefix + "login:token:"
	LoginTokenTTL    = 1800

	CacheShopKey     = Prefix + "cache:shop:"
	CacheShopTTL     = 30 // 分钟
	CacheShopLogicKey = Prefix + "cache:shop:logic:"
	LockShopKey      = Prefix + "lock:shop:"
	LockShopTTL      = 10 // 秒
	CacheNullTTL     = 2  // 分钟

	IDSeqKey         = Prefix + "id:seq:"

	SeckillStockKey  = Prefix + "seckill:stock:"
	SeckillOrderKey  = Prefix + "seckill:order:"
	SeckillStreamKey = Prefix + "stream:seckill:order"
	SeckillStreamGroup = "g1"

	RedpacketKey      = Prefix + "redpacket:"
	RedpacketAmounts  = ":amounts"
	RedpacketMeta      = ":meta"
	RedpacketTaken     = ":taken"
	RedpacketSeqKey    = Prefix + "redpacket:seq"
	RedpacketRateKey   = Prefix + "redpacket:rate:"
	RedpacketRecordStream = Prefix + "stream:redpacket:record"

	BlogLikedKey = Prefix + "blog:liked:"
	FollowKey    = Prefix + "follow:"
	FeedKey      = Prefix + "feed:"
	ShopGEOKey   = Prefix + "geo:shop"
	SignKey      = Prefix + "sign:"
	UVKey        = Prefix + "uv:"
	BloomShopKey = Prefix + "bloom:shop"
)
