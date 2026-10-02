<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { createRedPacket, grabRedPacket, redPacketRank } from '../api'

const router = useRouter()

const form = reactive({ title: '开学季红包雨', totalYuan: 100, count: 10 })
const currentRp = ref(null)
const rank = ref([])
const grabbing = ref(false)
const myAmount = ref(null)
const totalGot = ref(0) // 已被抢金额(分)
const remain = ref(0)  // 剩余个数
let timer = null

async function create() {
  const r = await createRedPacket(form.title, form.totalYuan, form.count)
  currentRp.value = r.data
  myAmount.value = null
  ElMessage.success(`红包雨已创建`)
  refreshRank()
}

async function grab() {
  if (!currentRp.value) return
  grabbing.value = true
  try {
    const r = await grabRedPacket(currentRp.value)
    myAmount.value = r.data
    ElMessage.success(`抢到 ${(r.data / 100).toFixed(2)} 元!`)
    refreshRank()
  } catch (e) {} finally {
    grabbing.value = false
  }
}

async function refreshRank() {
  if (!currentRp.value) return
  try {
    const r = await redPacketRank(currentRp.value)
    rank.value = r.data || []
    totalGot.value = rank.value.reduce((s, x) => s + Number(x.amount), 0)
  } catch (e) {}
}

onMounted(() => {
  timer = setInterval(refreshRank, 3000)
})
onUnmounted(() => clearInterval(timer))

function medal(i) {
  return ['🥇', '🥈', '🥉'][i] || (i + 1)
}
</script>

<template>
  <div class="page">
    <div class="page-title">红包雨</div>
    <div class="page-sub">创建红包雨场次,准时开抢,抢到就是赚到</div>

    <div class="rp-layout">
      <!-- 左:创建 + 抢 -->
      <div class="rp-left">
        <div class="card">
          <h3>创建红包雨</h3>
          <p class="muted" style="margin:8px 0">设置总金额和个数,系统自动拆分</p>
          <el-form :model="form" label-width="90px" style="margin-top:12px">
            <el-form-item label="标题">
              <el-input v-model="form.title" />
            </el-form-item>
            <el-form-item label="总金额(元)">
              <el-input-number v-model="form.totalYuan" :min="1" />
            </el-form-item>
            <el-form-item label="红包个数">
              <el-input-number v-model="form.count" :min="1" />
            </el-form-item>
            <el-form-item>
              <el-button type="danger" @click="create">创建红包雨</el-button>
            </el-form-item>
          </el-form>
        </div>

        <!-- 抢红包视觉中心 -->
        <div class="grab-card" v-if="currentRp">
          <div class="grab-bg"></div>
          <div class="grab-content">
            <div :class="['redpacket-icon', { shaking: grabbing }]">🧧</div>
            <div v-if="myAmount === null" class="grab-hint">
              <h2>{{ form.title }}</h2>
              <p>手慢无,赶紧抢!</p>
            </div>
            <div v-else class="grab-result">
              <p class="result-label">恭喜抢到</p>
              <div class="result-amount"><small>¥</small>{{ (myAmount / 100).toFixed(2) }}</div>
              <p class="result-tip">已放入账户</p>
            </div>
            <el-button type="danger" size="large" round :loading="grabbing"
                       :disabled="myAmount !== null" @click="grab" class="grab-btn">
              {{ myAmount !== null ? '已领取' : '立即抢红包' }}
            </el-button>
            <p class="grab-rule">每人限抢一次 · 10秒内最多操作3次</p>
          </div>
        </div>
      </div>

      <!-- 右:排行榜 -->
      <div class="rp-right">
        <div class="card" v-if="currentRp">
          <div class="card-title">
            <h3>🏆 领取榜</h3>
            <span class="muted">3秒刷新</span>
          </div>
          <div class="rp-stats">
            <div class="stat">
              <div class="stat-num">{{ rank.length }}</div>
              <div class="stat-label">已领</div>
            </div>
            <div class="stat">
              <div class="stat-num amount">¥{{ (totalGot / 100).toFixed(2) }}</div>
              <div class="stat-label">累计金额</div>
            </div>
          </div>
          <div class="rank-list">
            <div v-for="(r, i) in rank" :key="i" class="rank-item">
              <span class="rank-no">{{ medal(i) }}</span>
              <span class="rank-user link" @click="router.push(`/profile/${r.userId}`)">用户 {{ r.userId }} →</span>
              <span class="rank-amt">¥{{ (Number(r.amount) / 100).toFixed(2) }}</span>
            </div>
            <el-empty v-if="!rank.length" description="还没人领,快去抢!" :image-size="80" />
          </div>
        </div>
        <div class="card" v-else>
          <el-empty description="创建红包雨后开始抢" :image-size="100" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rp-layout { display: grid; grid-template-columns: 1fr 360px; gap: 16px; align-items: start; }

/* 抢红包卡片 */
.grab-card { position: relative; border-radius: 16px; overflow: hidden; min-height: 360px; }
.grab-bg { position: absolute; inset: 0; background: linear-gradient(160deg, #FF5A36 0%, #FF8A5C 50%, #FFB088 100%); }
.grab-bg::before { content: ''; position: absolute; right: -40px; top: -40px; width: 160px; height: 160px; border-radius: 50%; background: rgba(255,255,255,0.12); }
.grab-bg::after { content: ''; position: absolute; left: -30px; bottom: -50px; width: 120px; height: 120px; border-radius: 50%; background: rgba(255,255,255,0.08); }
.grab-content { position: relative; text-align: center; padding: 32px 20px; color: #fff; }
.redpacket-icon { font-size: 72px; line-height: 1; margin-bottom: 12px; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2)); }
.redpacket-icon.shaking { animation: shake 0.4s infinite; }
@keyframes shake { 0%,100% { transform: rotate(-6deg); } 50% { transform: rotate(6deg); } }
.grab-hint h2 { font-size: 22px; font-weight: 700; }
.grab-hint p { opacity: 0.9; margin-top: 4px; font-size: 14px; }
.grab-result { margin: 8px 0; }
.result-label { font-size: 14px; opacity: 0.9; }
.result-amount { font-size: 48px; font-weight: 700; line-height: 1.2; }
.result-amount small { font-size: 22px; }
.result-tip { font-size: 12px; opacity: 0.8; }
.grab-btn { margin-top: 20px; min-width: 200px; font-size: 16px; font-weight: 600; }
.grab-rule { margin-top: 14px; font-size: 11px; opacity: 0.75; }

/* 排行榜 */
.rp-stats { display: flex; gap: 12px; margin-bottom: 16px; }
.stat { flex: 1; background: var(--bg); border-radius: 10px; padding: 12px; text-align: center; }
.stat-num { font-size: 20px; font-weight: 700; }
.stat-label { font-size: 11px; color: var(--ink-3); margin-top: 2px; }
.rank-list { display: flex; flex-direction: column; gap: 8px; max-height: 360px; overflow-y: auto; }
.rank-item { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 8px; background: var(--bg); }
.rank-item:hover { background: var(--brand-light); }
.rank-no { width: 24px; text-align: center; font-weight: 700; color: var(--brand); }
.rank-user { flex: 1; font-size: 13px; color: var(--ink-2); }
.rank-user.link { cursor: pointer; }
.rank-user.link:hover { color: var(--brand); }
.rank-amt { font-weight: 700; color: var(--brand); font-size: 14px; }

@media (max-width: 900px) {
  .rp-layout { grid-template-columns: 1fr; }
}
</style>
