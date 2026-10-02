<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { seckill, querySeckillList, myOrders } from '../api'

const voucherId = ref(10)
const batch = reactive({ count: 1, interval: 50 })
const seckillVouchers = ref([])
const myOrderList = ref([])

// 秒杀结果持久化到 localStorage,切模块不丢
const LS_KEY = 'dss_seckill_results'
const results = ref(JSON.parse(localStorage.getItem(LS_KEY) || '[]'))
function saveResults() {
  localStorage.setItem(LS_KEY, JSON.stringify(results.value.slice(0, 50)))
}
function clearResults() {
  results.value = []
  localStorage.removeItem(LS_KEY)
  ElMessage.success('已清空记录')
}

const stats = computed(() => {
  const ok = results.value.filter(r => r.ok).length
  return { ok, fail: results.value.length - ok, total: results.value.length }
})

async function loadList() {
  try {
    const r = await querySeckillList(7)
    seckillVouchers.value = r.data || []
  } catch (e) {}
}

async function loadMyOrders() {
  try {
    const r = await myOrders()
    myOrderList.value = r.data || []
  } catch (e) {}
}

async function doOne(vid) {
  const id = vid || voucherId.value
  const t0 = performance.now()
  try {
    const r = await seckill(id)
    results.value.unshift({ ok: true, orderId: r.data, voucherId: id, cost: Math.round(performance.now() - t0), ts: Date.now() })
    saveResults()
    ElMessage.success(`秒杀成功 ${r.data}`)
    setTimeout(loadMyOrders, 1500) // 等异步落库后刷新
  } catch (e) {
    results.value.unshift({ ok: false, msg: e.msg || '失败', voucherId: id, cost: Math.round(performance.now() - t0), ts: Date.now() })
    saveResults()
  }
}

async function batchSeckill() {
  for (let i = 0; i < batch.count; i++) {
    doOne()
    if (batch.interval > 0) await new Promise(r => setTimeout(r, batch.interval))
  }
}

onMounted(() => { loadList(); loadMyOrders() })
</script>

<template>
  <div class="page">
    <div class="page-title">限时秒杀</div>
    <div class="page-sub">手快有手慢无,抢到的优惠券自动放入你的账户</div>

    <!-- 秒杀券展示 -->
    <div class="seckill-list">
      <div v-for="v in seckillVouchers" :key="v.id" class="sk-card">
        <div class="sk-glow"></div>
        <div class="sk-info">
          <h3>{{ v.title }}</h3>
          <p class="muted">{{ v.subTitle }}</p>
          <div class="sk-price">
            <span class="now"><small>¥</small>{{ (v.payValue / 100).toFixed(2) }}</span>
            <span class="orig">¥{{ (v.actualValue / 100).toFixed(2) }}</span>
            <el-tag type="danger" size="small" effect="dark">秒杀</el-tag>
          </div>
        </div>
        <el-button type="danger" size="large" round @click="voucherId = v.id; doOne()">立即抢</el-button>
      </div>
      <el-empty v-if="!seckillVouchers.length" description="暂无秒杀活动" />
    </div>

    <!-- 压测区 -->
    <div class="card">
      <div class="card-title">
        <h3>抢购测试</h3>
        <div class="stat-pills">
          <span class="pill ok">成功 {{ stats.ok }}</span>
          <span class="pill fail">失败 {{ stats.fail }}</span>
          <span class="pill">共 {{ stats.total }}</span>
        </div>
      </div>
      <div class="row" style="margin-top:8px">
        <span>秒杀券 ID:</span>
        <el-input-number v-model="voucherId" :min="1" />
        <el-button type="danger" @click="doOne">秒杀一次</el-button>
      </div>
      <el-divider />
      <div class="row">
        <span>并发次数:</span>
        <el-input-number v-model="batch.count" :min="1" :max="200" />
        <span>间隔(ms):</span>
        <el-input-number v-model="batch.interval" :min="0" :max="2000" />
        <el-button type="primary" @click="batchSeckill">批量秒杀</el-button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <h3>下单结果(本地保留,切页面不丢)</h3>
        <el-button size="small" @click="clearResults">清空</el-button>
      </div>
      <el-table :data="results" style="margin-top:12px" size="small" max-height="240">
        <el-table-column label="结果" width="80">
          <template #default="{ row }">
            <el-tag :type="row.ok ? 'success' : 'danger'" size="small">{{ row.ok ? '成功' : '失败' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="orderId" label="订单号" />
        <el-table-column prop="msg" label="失败原因" />
        <el-table-column prop="cost" label="耗时(ms)" width="100" />
      </el-table>
    </div>

    <div class="card">
      <div class="card-title">
        <h3>我的秒杀订单</h3>
        <el-button size="small" @click="loadMyOrders">刷新</el-button>
      </div>
      <el-table :data="myOrderList" size="small" max-height="240">
        <el-table-column prop="id" label="订单号" />
        <el-table-column prop="voucherId" label="券ID" width="80" />
        <el-table-column label="状态">
          <template #default="{ row }">
            <el-tag size="small" :type="row.status === 2 ? 'success' : 'info'">
              {{ {1:'未支付',2:'已支付',3:'已核销',4:'已取消'}[row.status] }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="下单时间" />
      </el-table>
      <el-empty v-if="!myOrderList.length" description="还没有订单,去抢一张" :image-size="60" />
    </div>
  </div>
</template>

<style scoped>
.seckill-list { margin-bottom: 16px; }
.sk-card { position: relative; display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 18px 20px; border-radius: 12px; background: linear-gradient(135deg, #FFF1EC 0%, #FFE0D4 100%); border: 1px solid #FFD0C0; margin-bottom: 12px; overflow: hidden; }
.sk-glow { position: absolute; right: -30px; top: -30px; width: 100px; height: 100px; border-radius: 50%; background: rgba(255,90,54,0.1); }
.sk-info { position: relative; }
.sk-info h3 { font-size: 17px; font-weight: 600; }
.sk-price { display: flex; align-items: baseline; gap: 10px; margin-top: 6px; }
.sk-price .now { color: var(--brand); font-size: 26px; font-weight: 700; }
.sk-price .now small { font-size: 15px; }
.sk-price .orig { color: var(--ink-3); text-decoration: line-through; font-size: 13px; }

.stat-pills { display: flex; gap: 6px; }
.pill { font-size: 12px; padding: 2px 10px; border-radius: 10px; background: var(--bg); color: var(--ink-3); }
.pill.ok { background: #E8F8EE; color: #00B42A; }
.pill.fail { background: #FFF0EC; color: var(--brand); }
</style>
