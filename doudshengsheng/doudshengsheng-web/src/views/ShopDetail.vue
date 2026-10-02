<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { queryShop, queryVoucherOfShop, querySeckillList, seckill } from '../api'
import { ElMessage } from 'element-plus'

const route = useRoute()
const router = useRouter()
const shop = ref(null)
const vouchers = ref([])
const seckills = ref([])
const strategy = ref('pass-through') // 默认标准加载,后端仍支持 ?strategy=mutex/logical 调用

// 我的券包(localStorage,演示)
const LS_KEY = 'dss_my_vouchers'
const showMyVouchers = ref(false)
const jumping = ref(false)
// claimedIds 只存 id(字符串),localStorage 里存的是对象数组,提取 id
const _saved = JSON.parse(localStorage.getItem(LS_KEY) || '[]')
const claimedIds = ref(new Set(_saved.map(x => String(x.id))))
const myVouchers = computed(() => JSON.parse(localStorage.getItem(LS_KEY) || '[]'))
function goSeckill() {
  jumping.value = true
  // 跳转后 watch route.params.id 会触发重新加载(组件复用不重建)
  router.push('/shop/7').finally(() => { jumping.value = false })
}
function claimVoucher(v) {
  const vid = String(v.id)
  if (claimedIds.value.has(vid)) {
    ElMessage.info('已领取过该券')
    return
  }
  claimedIds.value.add(vid)
  claimedIds.value = new Set(claimedIds.value) // 触发响应
  const list = JSON.parse(localStorage.getItem(LS_KEY) || '[]')
  list.push({ id: v.id, title: v.title, actualValue: v.actualValue, shopId: v.shopId, claimTime: Date.now() })
  localStorage.setItem(LS_KEY, JSON.stringify(list))
  ElMessage.success('领取成功,可在「我的券包」查看')
}

async function load() {
  const r = await queryShop(route.params.id, strategy.value)
  shop.value = r.data
}

async function loadAll() {
  await load()
  const sid = route.params.id
  const v = await queryVoucherOfShop(sid)
  vouchers.value = v.data || []
  const s = await querySeckillList(sid)
  seckills.value = s.data || []
}

onMounted(loadAll)

// 监听路由参数变化:从 /shop/1 跳 /shop/7 时组件复用,需重新加载数据
watch(() => route.params.id, (newId) => {
  if (newId) loadAll()
})

async function doSeckill(v) {
  try {
    const r = await seckill(v.id)
    ElMessage.success(`秒杀成功!订单号 ${r.data}`)
  } catch (e) {}
}

function stars(score) {
  const n = Math.round((score || 0) / 20)
  return '★'.repeat(n) + '☆'.repeat(5 - n)
}
</script>

<template>
  <div class="page">
    <!-- 头部大图 + 信息 -->
    <div class="shop-hero" v-if="shop">
      <img :src="shop.cover || `/covers/shop_${shop.id}.svg`" class="hero-img" />
      <div class="hero-mask"></div>
      <div class="hero-info">
        <h1>{{ shop.name }}</h1>
        <div class="hero-meta">
          <span class="rate">{{ stars(shop.score) }}</span>
          <span class="score">{{ (shop.score / 20).toFixed(1) }} 分</span>
          <span>销量 {{ shop.sold }}</span>
        </div>
        <p>{{ shop.area }} · {{ shop.address }}</p>
      </div>
    </div>

    <!-- 普通券 -->
    <div class="card">
      <div class="card-title">
        <h3>店铺优惠券</h3>
        <el-button text size="small" @click="showMyVouchers = true">我的券包 ({{ claimedIds.size }})</el-button>
      </div>
      <el-empty v-if="!vouchers.length" description="暂无优惠券" :image-size="80" />
      <div v-for="v in vouchers" :key="v.id" class="coupon">
        <div class="coupon-left">
          <div class="coupon-val"><small>¥</small>{{ (v.actualValue / 100).toFixed(0) }}</div>
          <div class="coupon-cond">满减券</div>
        </div>
        <div class="coupon-mid">
          <h4>{{ v.title }}</h4>
          <span class="muted">{{ v.subTitle }}</span>
        </div>
        <el-button type="primary" round :disabled="claimedIds.has(String(v.id))" @click="claimVoucher(v)">
          {{ claimedIds.has(String(v.id)) ? '已领取' : '领取' }}
        </el-button>
      </div>
    </div>

    <!-- 秒杀券 -->
    <div class="card">
      <div class="card-title">
        <h3>🔥 限时秒杀</h3>
        <el-tag type="danger" effect="plain" size="small">手快有</el-tag>
      </div>
      <el-empty v-if="!seckills.length" description="该店铺暂无秒杀活动" :image-size="80" />
      <div v-if="!seckills.length" style="text-align:center; margin-top:-8px">
        <el-button type="primary" size="small" round @click="goSeckill">
          {{ jumping ? '跳转中...' : '去看电影城秒杀 →' }}
        </el-button>
      </div>
      <div v-for="v in seckills" :key="v.id" class="seckill-card">
        <div class="seckill-glow"></div>
        <div class="seckill-body">
          <h4>{{ v.title }}</h4>
          <span class="muted">{{ v.subTitle }}</span>
          <div class="seckill-price">
            <span class="now"><small>¥</small>{{ (v.payValue / 100).toFixed(2) }}</span>
            <span class="orig">¥{{ (v.actualValue / 100).toFixed(2) }}</span>
          </div>
        </div>
        <el-button type="danger" size="large" round @click="doSeckill(v)">立即抢</el-button>
      </div>
    </div>

    <!-- 我的券包 -->
    <el-dialog v-model="showMyVouchers" title="我的券包" width="480px">
      <div v-for="v in myVouchers" :key="v.id" class="my-voucher">
        <div class="mv-val">¥{{ (v.actualValue / 100).toFixed(0) }}</div>
        <div class="mv-info">
          <div>{{ v.title }}</div>
          <span class="muted">领取时间:{{ new Date(v.claimTime).toLocaleString() }}</span>
        </div>
      </div>
      <el-empty v-if="!myVouchers.length" description="还没领取优惠券" />
    </el-dialog>
  </div>
</template>

<style scoped>
.my-voucher { display: flex; align-items: center; gap: 14px; padding: 12px; border: 1px dashed var(--line); border-radius: 8px; margin-bottom: 10px; }
.mv-val { color: var(--brand); font-size: 24px; font-weight: 700; min-width: 60px; }
.mv-info div { font-size: 14px; margin-bottom: 4px; }
.shop-hero { position: relative; border-radius: 14px; overflow: hidden; margin-bottom: 16px; height: 200px; }
.hero-img { width: 100%; height: 100%; object-fit: cover; }
.hero-mask { position: absolute; inset: 0; background: linear-gradient(0deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.1) 60%); }
.hero-info { position: absolute; left: 20px; bottom: 16px; color: #fff; }
.hero-info h1 { font-size: 24px; font-weight: 700; }
.hero-meta { display: flex; align-items: center; gap: 8px; font-size: 13px; margin: 4px 0; }
.hero-meta .rate { color: #FFD66E; letter-spacing: 1px; }
.hero-meta .score { font-weight: 600; }
.hero-info p { font-size: 12px; opacity: 0.9; }

.coupon { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px dashed var(--line); }
.coupon:last-child { border-bottom: none; }
.coupon-left { width: 90px; text-align: center; }
.coupon-val { color: var(--brand); font-size: 26px; font-weight: 700; }
.coupon-val small { font-size: 14px; }
.coupon-cond { font-size: 11px; color: var(--ink-3); }
.coupon-mid { flex: 1; }
.coupon-mid h4 { font-size: 14px; }

.seckill-card { position: relative; display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 16px; border-radius: 12px; background: linear-gradient(135deg, #FFF1EC 0%, #FFE8E0 100%); border: 1px solid #FFD0C0; margin-bottom: 12px; overflow: hidden; }
.seckill-glow { position: absolute; right: -20px; top: -20px; width: 80px; height: 80px; border-radius: 50%; background: rgba(255,90,54,0.1); }
.seckill-body { position: relative; }
.seckill-body h4 { font-size: 15px; font-weight: 600; }
.seckill-price { display: flex; align-items: baseline; gap: 8px; margin-top: 6px; }
.seckill-price .now { color: var(--brand); font-size: 22px; font-weight: 700; }
.seckill-price .now small { font-size: 13px; }
.seckill-price .orig { color: var(--ink-3); text-decoration: line-through; font-size: 13px; }
</style>
