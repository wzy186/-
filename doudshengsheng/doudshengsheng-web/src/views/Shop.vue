<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { queryShopTypeList, queryShopByType } from '../api'

const router = useRouter()
const types = ref([])
const shops = ref([])
const activeType = ref(null)

// 金刚位:分类 + 快捷入口
const quickEntries = [
  { id: 'seckill', label: '限时秒杀', icon: '⚡', color: '#FF5A36', path: '/voucher' },
  { id: 'redpacket', label: '红包雨', icon: '🧧', color: '#EE0979', path: '/redpacket' },
  { id: 'sign', label: '签到', icon: '📅', color: '#667eea', path: '/stats' },
  { id: 'blog', label: '探店', icon: '📝', color: '#43cea2', path: '/blog' },
  { id: 'follow', label: '关注', icon: '👥', color: '#4facfe', path: '/follow' },
  { id: 'nearby', label: '附近', icon: '📍', color: '#f5576c', path: '/nearby' }
]

onMounted(async () => {
  const r = await queryShopTypeList()
  types.value = r.data
  if (types.value.length) {
    activeType.value = types.value[0].id
    loadShops()
  }
})

async function loadShops() {
  const r = await queryShopByType(activeType.value)
  shops.value = r.data
}

function toDetail(id) {
  router.push(`/shop/${id}`)
}

function coverUrl(shop) {
  // 优先用后端 cover 字段(真实图),回退 SVG 占位
  return shop.cover || `/covers/shop_${shop.id}.svg`
}

// 评分转星级
function stars(score) {
  const n = Math.round((score || 0) / 20) // 0-100 -> 0-5
  return '★'.repeat(n) + '☆'.repeat(5 - n)
}
</script>

<template>
  <div class="home">
    <!-- Banner -->
    <div class="banner">
      <div class="banner-bg"></div>
      <div class="banner-content">
        <div class="banner-left">
          <h1>今日红包雨<span class="blink">即将开抢</span></h1>
          <p>抢到就是赚到,最高 100 元随手领</p>
          <el-button type="primary" size="large" round @click="router.push('/redpacket')">
            立即参与
          </el-button>
        </div>
        <div class="banner-card" @click="router.push('/voucher')">
          <div class="bc-tag">HOT</div>
          <div class="bc-title">9.9 元秒杀</div>
          <div class="bc-price"><small>¥</small>9<small>.9</small></div>
          <div class="bc-sub">电影票 · 限量抢</div>
        </div>
      </div>
    </div>

    <!-- 金刚位 -->
    <div class="quick">
      <div v-for="q in quickEntries" :key="q.id" class="quick-item" @click="router.push(q.path)">
        <div class="quick-icon" :style="{ background: q.color + '1f' }">{{ q.icon }}</div>
        <span>{{ q.label }}</span>
      </div>
    </div>

    <!-- 分类 + 商铺列表 -->
    <div class="section">
      <div class="section-head">
        <h2>优惠商铺</h2>
        <div class="cat-tabs">
          <span v-for="t in types" :key="t.id"
                :class="['cat-tab', { active: activeType === t.id }]"
                @click="activeType = t.id; loadShops()">
            {{ t.name }}
          </span>
        </div>
      </div>

      <div class="shop-grid">
        <div v-for="s in shops" :key="s.id" class="shop-card" @click="toDetail(s.id)">
          <div class="shop-cover">
            <img :src="coverUrl(s)" :alt="s.name" />
            <span class="shop-tag">优惠</span>
          </div>
          <div class="shop-body">
            <h3>{{ s.name }}</h3>
            <div class="shop-meta">
              <span class="shop-rate">{{ stars(s.score) }}</span>
              <span class="shop-score">{{ (s.score / 20).toFixed(1) }}</span>
              <span class="muted">销量 {{ s.sold }}</span>
            </div>
            <div class="shop-area">{{ s.area }} · {{ s.address }}</div>
            <div class="shop-foot">
              <span class="shop-price"><small>¥</small>{{ (s.avgPrice / 100).toFixed(0) }}<small>/人</small></span>
              <el-button type="primary" size="small" round>查看</el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.home { padding: 20px 24px 40px; max-width: 1200px; margin: 0 auto; }

/* Banner */
.banner { position: relative; border-radius: 16px; overflow: hidden; margin-bottom: 20px; }
.banner-bg { position: absolute; inset: 0; background: linear-gradient(120deg, #FF5A36 0%, #FF8A5C 60%, #FFB088 100%); }
.banner-bg::before { content: ''; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px; border-radius: 50%; background: rgba(255,255,255,0.12); }
.banner-bg::after { content: ''; position: absolute; left: 30%; bottom: -80px; width: 180px; height: 180px; border-radius: 50%; background: rgba(255,255,255,0.08); }
.banner-content { position: relative; display: flex; align-items: center; justify-content: space-between; padding: 32px 36px; }
.banner-left { color: #fff; }
.banner-left h1 { font-size: 28px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
.blink { font-size: 13px; background: #fff; color: var(--brand); padding: 3px 10px; border-radius: 12px; animation: blink 1.4s infinite; }
@keyframes blink { 50% { opacity: 0.5; } }
.banner-left p { margin: 8px 0 18px; opacity: 0.92; font-size: 14px; }
.banner-card { background: #fff; border-radius: 12px; padding: 16px 24px; text-align: center; cursor: pointer; box-shadow: 0 8px 24px rgba(0,0,0,0.15); transition: transform .2s; }
.banner-card:hover { transform: translateY(-3px); }
.bc-tag { background: var(--brand); color: #fff; font-size: 10px; padding: 1px 6px; border-radius: 4px; display: inline-block; }
.bc-title { font-size: 13px; color: var(--ink-3); margin: 4px 0; }
.bc-price { font-size: 30px; font-weight: 700; color: var(--brand); }
.bc-price small { font-size: 14px; }
.bc-sub { font-size: 11px; color: var(--ink-3); }

/* 金刚位 */
.quick { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 24px; background: #fff; padding: 20px; border-radius: 12px; box-shadow: var(--shadow-sm); }
.quick-item { display: flex; flex-direction: column; align-items: center; gap: 8px; cursor: pointer; }
.quick-item:hover .quick-icon { transform: scale(1.08); }
.quick-icon { width: 52px; height: 52px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 26px; transition: transform .2s; }
.quick-item span { font-size: 12px; color: var(--ink-2); }

/* 分类区 */
.section { }
.section-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.section-head h2 { font-size: 18px; font-weight: 600; }
.cat-tabs { display: flex; gap: 6px; }
.cat-tab { padding: 6px 14px; border-radius: 16px; font-size: 13px; color: var(--ink-2); cursor: pointer; transition: all .2s; }
.cat-tab:hover { background: var(--brand-light); }
.cat-tab.active { background: var(--brand); color: #fff; }

/* 商铺卡片 */
.shop-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.shop-card { background: #fff; border-radius: 12px; overflow: hidden; cursor: pointer; box-shadow: var(--shadow-sm); transition: all .2s; border: 1px solid var(--line); }
.shop-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }
.shop-cover { position: relative; height: 150px; overflow: hidden; }
.shop-cover img { width: 100%; height: 100%; object-fit: cover; }
.shop-tag { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.55); color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 4px; }
.shop-body { padding: 12px 14px; }
.shop-body h3 { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
.shop-meta { display: flex; align-items: center; gap: 6px; font-size: 12px; }
.shop-rate { color: var(--brand); letter-spacing: 1px; }
.shop-score { color: var(--brand); font-weight: 600; }
.shop-area { color: var(--ink-3); font-size: 12px; margin: 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.shop-foot { display: flex; align-items: center; justify-content: space-between; margin-top: 4px; }
.shop-price { color: var(--brand); font-weight: 700; font-size: 17px; }
.shop-price small { font-size: 11px; color: var(--ink-3); font-weight: 400; }

@media (max-width: 768px) {
  .quick { grid-template-columns: repeat(4, 1fr); }
  .shop-grid { grid-template-columns: 1fr; }
  .banner-content { flex-direction: column; gap: 16px; padding: 24px; }
  .banner-card { width: 100%; }
}
</style>
