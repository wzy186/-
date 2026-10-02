import request from '../utils/request'

// ===== 用户 =====
export const sendCode = (phone) => request.post(`/user/code?phone=${phone}`)
export const login = (data) => request.post('/user/login', data)
export const me = () => request.get('/user/me')
export const queryUser = (id) => request.get(`/user/${id}`)
export const userProfile = (id) => request.get(`/follow/profile/${id}`)
export const blogOfUser = (userId) => request.get(`/blog/of/user/${userId}`)

// ===== 商铺 =====
export const queryShop = (id, strategy) =>
  request.get(`/shop/${id}`, { params: { strategy } })
export const queryShopByType = (typeId) =>
  request.get('/shop/of/type', { params: { typeId } })
export const queryShopTypeList = () => request.get('/shop/type/list')
export const queryNearShop = (typeId, x, y, distKm) =>
  request.get('/shop/of/near', { params: { typeId, x, y, distKm } })

// ===== 优惠券秒杀 =====
export const queryVoucherOfShop = (shopId) => request.get(`/voucher/list/${shopId}`)
export const querySeckillList = (shopId) => request.get(`/voucher/seckill/list/${shopId}`)
export const seckill = (voucherId) => request.post(`/voucher/order/seckill/${voucherId}`)
export const myOrders = () => request.get('/voucher/order/my')

// ===== 红包雨 =====
export const createRedPacket = (title, totalYuan, count) =>
  request.post('/redpacket/create', null, { params: { title, totalYuan, count } })
export const grabRedPacket = (id) => request.post(`/redpacket/grab/${id}`)
export const redPacketRank = (id) => request.get(`/redpacket/rank/${id}`)

// ===== 博客点赞 =====
export const hotBlog = (current) => request.get('/blog/hot', { params: { current } })
export const likeBlog = (id) => request.put(`/blog/like/${id}`)
export const blogLikes = (id) => request.get(`/blog/likes/${id}`)
export const saveBlog = (data) => request.post('/blog', data)
export const blogDetail = (id) => request.get(`/blog/${id}`)

// ===== 关注 / Feed =====
export const follow = (id, isFollow) => request.put(`/follow/${id}/${isFollow}`)
export const isFollow = (id) => request.get(`/follow/or/${id}`)
export const commonFollows = (id) => request.get(`/follow/common/${id}`)
export const feed = (max, offset) => request.get('/follow/feed', { params: { max, offset } })

// ===== 签到 / UV =====
export const sign = () => request.post('/stats/sign')
export const signCount = () => request.get('/stats/sign/count')
export const signRecords = () => request.get('/stats/sign/records')
export const uv = (bizKey, userId) => request.post('/stats/uv', null, { params: { bizKey, userId } })
export const uvCount = (bizKey) => request.get('/stats/uv/count', { params: { bizKey } })

// ===== 商户后台 =====
export const adminAddVoucher = (data) => request.post('/admin/voucher', data)
export const adminVoucherStock = (id) => request.get(`/admin/voucher/stock/${id}`)
export const adminPreheat = (id) => request.post(`/admin/voucher/preheat/${id}`)
export const adminRedPacketList = () => request.get('/admin/redpacket/list')
export const adminRedPacketDetail = (id) => request.get(`/admin/redpacket/${id}`)
export const adminOrderStats = () => request.get('/admin/order/stats')

// ===== AI 助手 =====
// Agent 对话走 SSE。EventSource 不支持自定义 header,用 fetch + ReadableStream 实现。
export async function agentChatStream(query, token, { onStep, onToken, onDone, onError }) {
  const resp = await fetch('/api/ai/chat?query=' + encodeURIComponent(query), {
    method: 'POST',
    headers: { 'authorization': token, 'Accept': 'text/event-stream' }
  })
  if (!resp.ok) { onError('HTTP ' + resp.status); return }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    // SSE 事件以 \n\n 分隔
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      const lines = block.split('\n')
      let event = 'message', data = ''
      for (const line of lines) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (event === 'step') onStep(data)
      else if (event === 'token') onToken(data)
      else if (event === 'done') onDone(data)
      else if (event === 'error') onError(data)
    }
  }
}

// RAG 问答(同步)
export const ragAsk = (query) => request.post('/ai/rag', null, { params: { query } })
// 重建索引
export const ragReindex = () => request.post('/ai/reindex')
