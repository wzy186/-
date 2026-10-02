package handler

import (
	"context"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/cloudwego/hertz/pkg/route"
)

// API 接口定义(简易 Swagger 替代,不依赖外部生成工具)
type apiItem struct {
	Method      string `json:"method"`
	Path        string `json:"path"`
	Summary     string `json:"summary"`
	NeedLogin   bool   `json:"needLogin"`
	NeedAdmin   bool   `json:"needAdmin"`
}

var apiList = []apiItem{
	{"POST", "/user/code", "发送验证码", false, false},
	{"POST", "/user/login", "登录/注册", false, false},
	{"GET", "/user/me", "当前用户", true, false},
	{"GET", "/user/:id", "查用户", false, false},
	{"GET", "/shop/:id", "商铺详情(三策略)", false, false},
	{"PUT", "/shop", "更新商铺", false, true},
	{"POST", "/shop", "新增商铺", false, true},
	{"GET", "/shop/of/type", "按类型查商铺", false, false},
	{"GET", "/shop/of/near", "附近商铺(GEO)", false, false},
	{"GET", "/shop/type/list", "商铺类型", false, false},
	{"GET", "/voucher/list/:shopId", "商铺优惠券", false, false},
	{"GET", "/voucher/seckill/list/:shopId", "商铺秒杀券", false, false},
	{"POST", "/voucher/order/seckill/:id", "秒杀下单", true, false},
	{"GET", "/voucher/order/my", "我的订单", true, false},
	{"GET", "/admin/voucher/stock/:id", "秒杀券库存", false, true},
	{"POST", "/admin/voucher/preheat/:id", "预热库存", false, true},
	{"POST", "/redpacket/create", "创建红包雨", true, false},
	{"POST", "/redpacket/grab/:id", "抢红包", true, false},
	{"GET", "/redpacket/rank/:id", "领取排行榜", true, false},
	{"GET", "/admin/redpacket/list", "红包雨场次列表", false, true},
	{"GET", "/admin/redpacket/:id", "红包雨场次详情", false, true},
	{"GET", "/blog/hot", "热门笔记", false, false},
	{"GET", "/blog/:id", "笔记详情", false, false},
	{"PUT", "/blog/like/:id", "点赞/取消", true, false},
	{"GET", "/blog/likes/:id", "点赞榜", false, false},
	{"POST", "/blog", "发布笔记", true, false},
	{"GET", "/blog/of/user/:userId", "用户笔记", false, false},
	{"PUT", "/follow/:id/:isFollow", "关注/取关", true, false},
	{"GET", "/follow/or/:id", "是否关注", true, false},
	{"GET", "/follow/common/:id", "共同关注", true, false},
	{"GET", "/follow/feed", "Feed流", true, false},
	{"GET", "/follow/profile/:id", "用户主页", false, false},
	{"POST", "/stats/sign", "签到", true, false},
	{"GET", "/stats/sign/count", "连续签到", true, false},
	{"GET", "/stats/sign/records", "本月签到记录", true, false},
	{"POST", "/stats/uv", "记录UV", false, false},
	{"GET", "/stats/uv/count", "UV数", false, false},
	{"POST", "/ai/chat", "Agent对话(SSE)", true, false},
	{"POST", "/ai/rag", "RAG问答", true, false},
	{"POST", "/ai/reindex", "重建索引", true, false},
}

type DocHandler struct{}

func NewDocHandler() *DocHandler { return &DocHandler{} }

func (h *DocHandler) Register(group *route.RouterGroup) {
	group.GET("/swagger", h.Page)
	group.GET("/swagger/apis", h.APIs)
}

func (h *DocHandler) APIs(c context.Context, ctx *app.RequestContext) {
	ctx.JSON(consts.StatusOK, map[string]interface{}{
		"title": "兜省省 Go 版 API",
		"count": len(apiList),
		"apis":  apiList,
	})
}

func (h *DocHandler) Page(c context.Context, ctx *app.RequestContext) {
	ctx.SetContentType("text/html; charset=utf-8")
	ctx.SetBodyString(swaggerHTML)
}

const swaggerHTML = `<!doctype html><html><head><meta charset="utf-8">
<title>兜省省 Go 版 API 文档</title>
<style>
body{font-family:-apple-system,"PingFang SC",sans-serif;background:#f5f7fa;margin:0;padding:20px;color:#1a1a1a}
h1{color:#FF5A36}.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;color:#fff;margin-right:8px;width:50px;text-align:center}
.get{background:#4facfe}.post{background:#52c41a}.put{background:#faad14}.del{background:#ff4d4f}
table{background:#fff;border-radius:8px;overflow:hidden;width:100%;border-collapse:collapse;box-shadow:0 1px 3px rgba(0,0,0,.06)}
td,th{padding:10px 14px;text-align:left;border-bottom:1px solid #f0f0f0}
th{background:#fafafa}.lock{color:#faad14;font-size:12px}.admin{color:#ff4d4f;font-size:12px}
input{padding:6px 10px;border:1px solid #ddd;border-radius:4px;width:200px}
</style></head><body>
<h1>💰 兜省省 Go 版 API 文档</h1>
<p>共 <b id="cnt">-</b> 个接口 · <span class="lock">🔒需登录</span> · <span class="admin">🔐仅管理员</span></p>
<input placeholder="搜索..." oninput="filter(this.value)">
<table id="tbl"><thead><tr><th>方法</th><th>路径</th><th>说明</th><th>权限</th></tr></thead><tbody id="tb"></tbody></table>
<script>
fetch('/swagger/apis').then(r=>r.json()).then(d=>{document.getElementById('cnt').textContent=d.count;render(d.apis)})
function render(apis){
  const tb=document.getElementById('tb');tb.innerHTML='';
  apis.forEach(a=>{const m=a.method.toLowerCase();
    const tr=document.createElement('tr');tr.dataset.text=(a.method+a.path+a.summary).toLowerCase();
    const perm=a.needAdmin?'<span class="admin">🔐管理员</span>':a.needLogin?'<span class="lock">🔒登录</span>':'-';
    tr.innerHTML='<td><span class="tag '+m+'">'+a.method+'</span></td><td><code>'+a.path+'</code></td><td>'+a.summary+'</td><td>'+perm+'</td>';
    tb.appendChild(tr)})}
function filter(q){document.querySelectorAll('#tb tr').forEach(tr=>{tr.style.display=tr.dataset.text.includes(q.toLowerCase())?'':'none'})}
</script></body></html>`
