package ai

import (
	"context"
	"fmt"
	"log"
	"sync"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"

	"gorm.io/gorm"
)

// Tool AI 工具接口(对应 Java AiTool),LLM 可调
type Tool interface {
	Name() string
	Desc() string
	Params() map[string]interface{}
	Execute(ctx context.Context, args map[string]interface{}) string
}

// ToolRegistry 工具注册中心(对应 Java ToolRegistry)
type ToolRegistry struct {
	tools map[string]Tool
	mu    sync.RWMutex
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: map[string]Tool{}}
}

func (r *ToolRegistry) Register(t Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[t.Name()] = t
	log.Println("注册 AI 工具:", t.Name())
}

// AllDefs 所有工具定义(给 LLM)
func (r *ToolRegistry) AllDefs() []ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()
	defs := []ToolDef{}
	for _, t := range r.tools {
		defs = append(defs, ToolDef{
			Type: "function",
			Function: ToolFunction{
				Name: t.Name(), Description: t.Desc(), Parameters: t.Params(),
			},
		})
	}
	return defs
}

// Execute 执行工具
func (r *ToolRegistry) Execute(ctx context.Context, name string, args map[string]interface{}) string {
	r.mu.RLock()
	t, ok := r.tools[name]
	r.mu.RUnlock()
	if !ok {
		return "工具不存在: " + name
	}
	return t.Execute(ctx, args)
}

// --- 5 个工具实现 ---

// SearchShopTool 搜索商铺
type SearchShopTool struct{}

func (SearchShopTool) Name() string { return "search_shops" }
func (SearchShopTool) Desc() string {
	return "搜索商铺。可按类型(美食/娱乐/丽人/生活服务/酒店)或关键词搜索。返回商铺名、地址、均价、评分、销量。"
}
func (SearchShopTool) Params() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"type":    map[string]interface{}{"type": "string", "description": "商铺类型"},
			"keyword": map[string]interface{}{"type": "string", "description": "商铺名关键词"},
		},
	}
}
func (SearchShopTool) Execute(ctx context.Context, args map[string]interface{}) string {
	q := utils.DB.Model(&model.Shop{})
	if kw, ok := args["keyword"].(string); ok && kw != "" {
		q = q.Where("name LIKE ?", "%"+kw+"%")
	}
	if t, ok := args["type"].(string); ok && t != "" {
		var st model.ShopType
		if utils.DB.Where("name LIKE ?", "%"+t+"%").First(&st).Error == nil {
			q = q.Where("type_id = ?", st.ID)
		}
	}
	var shops []model.Shop
	q.Limit(10).Find(&shops)
	if len(shops) == 0 {
		return "没有找到匹配的商铺"
	}
	out := ""
	for _, s := range shops {
		out += fmt.Sprintf("- %s(地址:%s,均价%d元/人,评分%.1f,销量%d)\n",
			s.Name, s.Address, s.AvgPrice/100, float64(s.Score)/20.0, s.Sold)
	}
	return out
}

// ListVouchersTool 查优惠券
type ListVouchersTool struct{}

func (ListVouchersTool) Name() string { return "list_vouchers" }
func (ListVouchersTool) Desc() string {
	return "查询所有优惠券(含秒杀券)。用户问有什么优惠时调用。"
}
func (ListVouchersTool) Params() map[string]interface{} {
	return map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
}
func (ListVouchersTool) Execute(ctx context.Context, args map[string]interface{}) string {
	var vs []model.Voucher
	utils.DB.Limit(20).Find(&vs)
	if len(vs) == 0 {
		return "暂无优惠券"
	}
	out := ""
	for _, v := range vs {
		tag := ""
		if v.Type == 2 {
			tag = ",秒杀券"
		}
		out += fmt.Sprintf("- %s(抵扣%d元%s)\n", v.Title, v.ActualValue/100, tag)
	}
	return out
}

// ListRedPacketsTool 查进行中的红包雨
type ListRedPacketsTool struct{}

func (ListRedPacketsTool) Name() string { return "list_redpackets" }
func (ListRedPacketsTool) Desc() string {
	return "查询进行中的红包雨场次。用户问'有什么红包可以抢'时调用,返回场次ID、标题、剩余个数。"
}
func (ListRedPacketsTool) Params() map[string]interface{} {
	return map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
}
func (ListRedPacketsTool) Execute(ctx context.Context, args map[string]interface{}) string {
	var rps []model.RedPacket
	utils.DB.Where("status = 1").Find(&rps)
	if len(rps) == 0 {
		return "暂无进行中的红包雨"
	}
	out := ""
	for _, rp := range rps {
		out += fmt.Sprintf("- 场次ID:%d,标题:%s,剩余%d个,总额%.2f元\n",
			rp.ID, rp.Title, rp.RemainCount, float64(rp.TotalAmount)/100)
	}
	return out
}

// SeckillTool 秒杀下单
type SeckillTool struct {
	Svc interface {
		Seckill(ctx context.Context, voucherID, userID int64) *utils.Result
	}
}

func (SeckillTool) Name() string { return "seckill_voucher" }
func (SeckillTool) Desc() string {
	return "对秒杀券下单(秒杀)。需要券ID。用户说'帮我秒杀/抢电影票'时调用。"
}
func (SeckillTool) Params() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"voucherId": map[string]interface{}{"type": "integer", "description": "秒杀券ID"},
		},
		"required": []string{"voucherId"},
	}
}
func (t SeckillTool) Execute(ctx context.Context, args map[string]interface{}) string {
	uid := utils.GetUserID(ctx)
	if uid == 0 {
		return "需要先登录"
	}
	vid := toInt64FromArgs(args["voucherId"])
	r := t.Svc.Seckill(ctx, vid, uid)
	if r.Success {
		return fmt.Sprintf("秒杀成功!订单号: %v", r.Data)
	}
	return "秒杀失败: " + r.Msg
}

// GrabRedPacketTool 抢红包
type GrabRedPacketTool struct {
	Svc interface {
		Grab(ctx context.Context, rpID, userID int64) *utils.Result
	}
}

func (GrabRedPacketTool) Name() string { return "grab_redpacket" }
func (GrabRedPacketTool) Desc() string {
	return "抢红包雨。需要红包场次ID。用户说'帮我抢红包'时调用。"
}
func (GrabRedPacketTool) Params() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"redPacketId": map[string]interface{}{"type": "integer", "description": "红包场次ID"},
		},
		"required": []string{"redPacketId"},
	}
}
func (t GrabRedPacketTool) Execute(ctx context.Context, args map[string]interface{}) string {
	uid := utils.GetUserID(ctx)
	if uid == 0 {
		return "需要先登录"
	}
	rpID := toInt64FromArgs(args["redPacketId"])
	r := t.Svc.Grab(ctx, rpID, uid)
	if r.Success {
		amt := toInt64FromArgs(r.Data)
		return fmt.Sprintf("抢到红包!金额: %.2f 元", float64(amt)/100)
	}
	return "抢红包失败: " + r.Msg
}

func toInt64FromArgs(v interface{}) int64 {
	if v == nil {
		return 0
	}
	var i int64
	switch x := v.(type) {
	case float64:
		i = int64(x)
	case string:
		fmt.Sscanf(x, "%d", &i)
	default:
		fmt.Sscanf(fmt.Sprintf("%v", v), "%d", &i)
	}
	return i
}

var _ = gorm.ErrRecordNotFound
