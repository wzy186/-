<script setup>
import { ref, onMounted } from 'vue'
import { adminOrderStats, adminVoucherStock, adminRedPacketList } from '../../api'

const orderStats = ref(null)
const stock = ref(null)
const rpCount = ref(0)
const loading = ref(false)

async function load() {
  loading.value = true
  const [o, s, r] = await Promise.all([
    adminOrderStats(),
    adminVoucherStock(10),
    adminRedPacketList()
  ])
  orderStats.value = o.data
  stock.value = s.data
  rpCount.value = (r.data || []).length
  loading.value = false
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-title">数据概览</div>
    <div class="page-sub">实时业务数据,每刷新一次更新</div>

    <div class="kpis">
      <div class="kpi">
        <div class="kpi-icon" style="background:#FFF1EC">📦</div>
        <div>
          <div class="kpi-num">{{ orderStats?.total || 0 }}</div>
          <div class="kpi-label">秒杀订单总数</div>
        </div>
      </div>
      <div class="kpi">
        <div class="kpi-icon" style="background:#E8F8EE">🎫</div>
        <div>
          <div class="kpi-num">{{ stock?.redisStock ?? '-' }}</div>
          <div class="kpi-label">秒杀券剩余库存</div>
        </div>
      </div>
      <div class="kpi">
        <div class="kpi-icon" style="background:#FFF6E6">🧧</div>
        <div>
          <div class="kpi-num">{{ rpCount }}</div>
          <div class="kpi-label">红包雨场次</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <h3>各券订单数</h3>
        <el-button size="small" @click="load">刷新</el-button>
      </div>
      <el-table :data="orderStats?.byVoucher || []" size="small">
        <el-table-column prop="voucher_id" label="秒杀券 ID" />
        <el-table-column prop="cnt" label="订单数" />
      </el-table>
      <el-empty v-if="!orderStats?.byVoucher?.length" description="暂无订单" />
    </div>

    <div class="card">
      <h3>秒杀券库存对比</h3>
      <p class="muted" style="margin:8px 0">Redis 实时库存 vs DB 初始库存</p>
      <div class="row" v-if="stock">
        <el-tag size="large">Redis: {{ stock.redisStock }}</el-tag>
        <el-tag size="large" type="info">DB: {{ stock.dbStock }}</el-tag>
        <el-progress :percentage="stock.dbStock ? Math.round(stock.redisStock / stock.dbStock * 100) : 0"
                     :stroke-width="12" style="flex:1; max-width:300px" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.kpis { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:20px; }
.kpi { background:#fff; border-radius:12px; padding:20px; display:flex; align-items:center; gap:16px; box-shadow:var(--shadow-sm); border:1px solid var(--line); }
.kpi-icon { width:52px; height:52px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px; }
.kpi-num { font-size:26px; font-weight:700; }
.kpi-label { font-size:12px; color:var(--ink-3); margin-top:2px; }
</style>
