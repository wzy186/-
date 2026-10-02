<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { sign, signCount, signRecords, uv, uvCount } from '../api'

const signDays = ref(0)
const signedToday = ref(false)
const signRecs = ref([false]) // 下标=日期,0占位
const selectedDay = ref(null)
const uvBizKey = ref('homepage')
const uvNum = ref(0)
const uvForm = reactive({ bizKey: 'homepage', userId: null })
const today = new Date()
const monthDays = computed(() => new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate())
const todayDate = today.getDate()

function isSigned(d) { return signRecs.value[d] === true }

async function doSign() {
  if (signedToday.value) { ElMessage.info('今天已签到'); return }
  await sign()
  signedToday.value = true
  ElMessage.success('签到成功 +1')
  loadSignCount()
}

async function loadSignCount() {
  const r = await signCount()
  signDays.value = r.data
  if (r.data > 0) signedToday.value = true
  // 加载本月签到记录,日历按真实记录标已签
  const rr = await signRecords()
  signRecs.value = rr.data || [false]
}

async function addUv() {
  const uid = uvForm.userId || Math.floor(Math.random() * 1000)
  await uv(uvForm.bizKey, uid)
  ElMessage.success(`访客 ${uid} 已记录`)
  loadUvCount()
}

async function loadUvCount() {
  const r = await uvCount(uvBizKey.value)
  uvNum.value = r.data
}

onMounted(() => { loadSignCount(); loadUvCount() })
</script>

<template>
  <div class="page">
    <div class="page-title">签到中心</div>
    <div class="page-sub">每日签到攒连续天数,查看页面访客统计</div>

    <div class="stats-layout">
      <!-- 签到卡 -->
      <div class="sign-card">
        <div class="sign-bg"></div>
        <div class="sign-content">
          <div class="sign-ring">
            <div class="ring-num">{{ signDays }}</div>
            <div class="ring-label">连续签到</div>
          </div>
          <h3>坚持就是省钱</h3>
          <p class="muted" style="color:rgba(255,255,255,0.85)">已连续签到 {{ signDays }} 天,继续保持</p>
          <el-button type="danger" size="large" round :disabled="signedToday" @click="doSign" class="sign-btn">
            {{ signedToday ? '今日已签' : '立即签到' }}
          </el-button>
        </div>
      </div>

      <!-- 本月日历 -->
      <div class="card">
        <div class="card-title">
          <h3>本月日历</h3>
          <span class="muted">{{ today.getMonth() + 1 }} 月</span>
        </div>
        <div class="cal-grid">
          <div v-for="d in monthDays" :key="d"
               :class="['cal-day', { today: d === todayDate, signed: isSigned(d), selected: selectedDay === d }]"
               @click="selectedDay = d">
            {{ d }}
          </div>
        </div>
        <p class="muted" style="margin-top:8px; font-size:12px" v-if="selectedDay">
          {{ today.getMonth()+1 }}月{{ selectedDay }}日:
          <span v-if="selectedDay > todayDate">尚未到来</span>
          <span v-else-if="isSigned(selectedDay)" class="amount">已签到 ✓</span>
          <span v-else>未签到</span>
        </p>
        <div class="cal-legend">
          <span><i class="dot signed"></i> 已签</span>
          <span><i class="dot today"></i> 今天</span>
        </div>
      </div>
    </div>

    <!-- UV 统计 -->
    <div class="card">
      <div class="card-title">
        <h3>访客统计</h3>
        <span class="muted">去重计数</span>
      </div>
      <div class="uv-layout">
        <div class="uv-display">
          <div class="uv-num">{{ uvNum }}</div>
          <div class="uv-label">独立访客</div>
        </div>
        <div class="uv-form">
          <div class="row">
            <span>业务:</span>
            <el-input v-model="uvForm.bizKey" style="width:140px" />
            <span>访客ID:</span>
            <el-input-number v-model="uvForm.userId" :min="0" placeholder="随机" />
            <el-button type="primary" @click="addUv">记录访客</el-button>
            <el-button @click="loadUvCount">刷新</el-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.stats-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

.sign-card { position: relative; border-radius: 14px; overflow: hidden; min-height: 280px; }
.sign-bg { position: absolute; inset: 0; background: linear-gradient(150deg, #FF5A36 0%, #FF8A5C 100%); }
.sign-bg::before { content: ''; position: absolute; right: -40px; top: -40px; width: 160px; height: 160px; border-radius: 50%; background: rgba(255,255,255,0.12); }
.sign-content { position: relative; padding: 28px; text-align: center; color: #fff; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.sign-ring { width: 110px; height: 110px; border-radius: 50%; border: 4px solid rgba(255,255,255,0.4); display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.1); }
.ring-num { font-size: 36px; font-weight: 700; line-height: 1; }
.ring-label { font-size: 11px; opacity: 0.9; margin-top: 2px; }
.sign-content h3 { font-size: 16px; }
.sign-btn { margin-top: 8px; min-width: 160px; }

.cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; }
.cal-day { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-size: 12px; color: var(--ink-3); background: var(--bg); cursor: pointer; }
.cal-day:hover { background: var(--brand-light); }
.cal-day.selected { outline: 2px solid var(--brand); outline-offset: -2px; }
.cal-day.signed { background: var(--brand); color: #fff; }
.cal-day.today { border: 2px solid var(--brand); color: var(--brand); font-weight: 700; }
.cal-day.today.signed { color: #fff; }
.cal-legend { display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: var(--ink-3); }
.cal-legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }
.cal-legend .dot.signed { background: var(--brand); }
.cal-legend .dot.today { border: 2px solid var(--brand); }

.uv-layout { display: flex; align-items: center; gap: 24px; }
.uv-display { text-align: center; min-width: 140px; }
.uv-num { font-size: 42px; font-weight: 700; color: var(--brand); line-height: 1; }
.uv-label { font-size: 12px; color: var(--ink-3); margin-top: 4px; }
.uv-form { flex: 1; }

@media (max-width: 768px) {
  .stats-layout { grid-template-columns: 1fr; }
  .uv-layout { flex-direction: column; }
}
</style>
