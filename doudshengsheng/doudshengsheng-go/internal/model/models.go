package model

import "time"

// User 用户表(对应 tb_user)
type User struct {
	ID        int64     `gorm:"column:id;primaryKey" json:"id"`
	Phone     string    `gorm:"column:phone" json:"phone"`
	Password  string    `gorm:"column:password" json:"-"`
	NickName  string    `gorm:"column:nick_name" json:"nickName"`
	Icon      string    `gorm:"column:icon" json:"icon"`
	Role      int       `gorm:"column:role" json:"role"` // 0普通 1管理员
	CreatedAt time.Time `gorm:"column:create_time" json:"createTime"`
	UpdatedAt time.Time `gorm:"column:update_time" json:"updateTime"`
}

func (User) TableName() string { return "tb_user" }

// ShopType 商铺类型
type ShopType struct {
	ID   int64  `gorm:"column:id;primaryKey" json:"id"`
	Name string `gorm:"column:name" json:"name"`
	Sort int    `gorm:"column:sort" json:"sort"`
}

func (ShopType) TableName() string { return "tb_shop_type" }

// Shop 商铺
type Shop struct {
	ID        int64     `gorm:"column:id;primaryKey" json:"id"`
	Name      string    `gorm:"column:name" json:"name"`
	TypeID    int64     `gorm:"column:type_id" json:"typeId"`
	Images    string    `gorm:"column:images" json:"images"`
	Cover     string    `gorm:"column:cover" json:"cover"`
	Area      string    `gorm:"column:area" json:"area"`
	Address   string    `gorm:"column:address" json:"address"`
	X         float64   `gorm:"column:x" json:"x"`
	Y         float64   `gorm:"column:y" json:"y"`
	AvgPrice  int64     `gorm:"column:avg_price" json:"avgPrice"`
	Sold      int       `gorm:"column:sold" json:"sold"`
	Comments  int       `gorm:"column:comments" json:"comments"`
	Score     int       `gorm:"column:score" json:"score"`
	OpenHours string    `gorm:"column:open_hours" json:"openHours"`
	CreatedAt time.Time `gorm:"column:create_time" json:"createTime"`
	UpdatedAt time.Time `gorm:"column:update_time" json:"updateTime"`
}

func (Shop) TableName() string { return "tb_shop" }

// Voucher 优惠券
type Voucher struct {
	ID          int64     `gorm:"column:id;primaryKey" json:"id"`
	ShopID      *int64    `gorm:"column:shop_id" json:"shopId"`
	Title       string    `gorm:"column:title" json:"title"`
	SubTitle    string    `gorm:"column:sub_title" json:"subTitle"`
	Rules       string    `gorm:"column:rules" json:"rules"`
	PayValue    int64     `gorm:"column:pay_value" json:"payValue"`
	ActualValue int64     `gorm:"column:actual_value" json:"actualValue"`
	Type        int       `gorm:"column:type" json:"type"`     // 1普通 2秒杀
	Status      int       `gorm:"column:status" json:"status"` // 1上架 0下架
	CreatedAt   time.Time `gorm:"column:create_time" json:"createTime"`
}

func (Voucher) TableName() string { return "tb_voucher" }

// SeckillVoucher 秒杀券附加信息
type SeckillVoucher struct {
	VoucherID  int64     `gorm:"column:voucher_id;primaryKey" json:"voucherId"`
	Stock      int       `gorm:"column:stock" json:"stock"`
	BeginTime  time.Time `gorm:"column:begin_time" json:"beginTime"`
	EndTime    time.Time `gorm:"column:end_time" json:"endTime"`
	CreatedAt  time.Time `gorm:"column:create_time" json:"createTime"`
}

func (SeckillVoucher) TableName() string { return "tb_seckill_voucher" }

// VoucherOrder 优惠券订单
type VoucherOrder struct {
	ID        int64     `gorm:"column:id;primaryKey" json:"id"` // 全局唯一ID
	UserID    int64     `gorm:"column:user_id" json:"userId"`
	VoucherID int64     `gorm:"column:voucher_id" json:"voucherId"`
	PayType   int       `gorm:"column:pay_type" json:"payType"`
	Status    int       `gorm:"column:status" json:"status"` // 1未支付 2已支付 3已核销 4已取消
	CreatedAt time.Time `gorm:"column:create_time" json:"createTime"`
}

func (VoucherOrder) TableName() string { return "tb_voucher_order" }

// Blog 探店笔记
type Blog struct {
	ID        int64     `gorm:"column:id;primaryKey" json:"id"`
	ShopID    *int64    `gorm:"column:shop_id" json:"shopId"`
	UserID    int64     `gorm:"column:user_id" json:"userId"`
	Title     string    `gorm:"column:title" json:"title"`
	Content   string    `gorm:"column:content" json:"content"`
	Images    string    `gorm:"column:images" json:"images"`
	Liked     int       `gorm:"column:liked" json:"liked"`
	Comments  int       `gorm:"column:comments" json:"comments"`
	CreatedAt time.Time `gorm:"column:create_time" json:"createTime"`
	UpdatedAt time.Time `gorm:"column:update_time" json:"updateTime"`
}

func (Blog) TableName() string { return "tb_blog" }

// Follow 关注关系
type Follow struct {
	ID           int64     `gorm:"column:id;primaryKey" json:"id"`
	UserID       int64     `gorm:"column:user_id" json:"userId"`
	FollowUserID int64     `gorm:"column:follow_user_id" json:"followUserId"`
	CreatedAt    time.Time `gorm:"column:create_time" json:"createTime"`
}

func (Follow) TableName() string { return "tb_follow" }

// RedPacket 红包雨场次
type RedPacket struct {
	ID          int64     `gorm:"column:id;primaryKey" json:"id"`
	Title       string    `gorm:"column:title" json:"title"`
	TotalAmount int64     `gorm:"column:total_amount" json:"totalAmount"` // 分
	Count       int       `gorm:"column:count" json:"count"`
	RemainCount int       `gorm:"column:remain_count" json:"remainCount"`
	GotCount    int       `gorm:"column:got_count" json:"gotCount"`
	Status      int       `gorm:"column:status" json:"status"` // 1进行中 2已抢完 3已退款
	CreatedAt   time.Time `gorm:"column:create_time" json:"createTime"`
	EndTime     *time.Time `gorm:"column:end_time" json:"endTime"`
}

func (RedPacket) TableName() string { return "tb_red_packet" }

// RedPacketRecord 红包领取记录
type RedPacketRecord struct {
	ID          int64     `gorm:"column:id;primaryKey" json:"id"`
	RedPacketID int64     `gorm:"column:red_packet_id" json:"redPacketId"`
	UserID      int64     `gorm:"column:user_id" json:"userId"`
	Amount      int64     `gorm:"column:amount" json:"amount"`
	GrabTime    time.Time `gorm:"column:grab_time" json:"grabTime"`
}

func (RedPacketRecord) TableName() string { return "tb_red_packet_record" }
