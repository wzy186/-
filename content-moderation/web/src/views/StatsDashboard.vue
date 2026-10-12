<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { api } from '../api'
import { ACTION_LABEL, ACTION_TAG, STATUS_COLOR, STATUS_LABEL, fmtTime } from '../constants'
import type { Stats } from '../types'

const router = useRouter()
const stats = ref<Stats | null>(null)
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    stats.value = await api.stats()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : '加载失败')
  } finally {
    loading.value = false
  }
}

const rateText = (r: number | null) => (r === null ? '—' : `${r}%`)

const kpis = computed(() => {
  const s = stats.value
  return [
    {
      label: '待审核积压',
      icon: '⏳',
      color: '#e6a23c',
      tint: '#fdf6ec',
      value: s ? String(s.pendingCount) : '—',
      caption: '条投稿等待处理',
    },
    {
      label: '今日已处理',
      icon: '✅',
      color: '#67c23a',
      tint: '#f0f9eb',
      value: s ? String(s.today.processed) : '—',
      caption: s ? `通过 ${s.today.approved} · 驳回 ${s.today.rejected}` : '',
    },
    {
      label: '今日通过率',
      icon: '📈',
      color: '#409eff',
      tint: '#ecf5ff',
      value: s ? rateText(s.today.approveRate) : '—',
      caption: '今日通过 / 已处理',
    },
    {
      label: '累计通过率',
      icon: '🗂️',
      color: '#409eff',
      tint: '#ecf5ff',
      value: s ? rateText(s.overall.approveRate) : '—',
      caption: s ? `累计处理 ${s.overall.processed} 条` : '',
    },
  ]
})

// 近 7 日分组柱状图数据
const chart = computed(() => {
  const days = stats.value?.last7Days ?? []
  const max = Math.max(1, ...days.map((d) => d.approved + d.rejected))
  return { days, max }
})
const barHeight = (v: number) => `${(v / chart.value.max) * 100}%`

// 状态分布堆叠条数据（文案与颜色取自共享常量，与队列页标签一致）
const distribution = computed(() => {
  const d = stats.value?.statusDistribution ?? { pending: 0, approved: 0, rejected: 0 }
  const total = Math.max(1, d.pending + d.approved + d.rejected)
  const defs = (['pending', 'approved', 'rejected'] as const).map((key) => ({
    key,
    label: STATUS_LABEL[key],
    color: STATUS_COLOR[key],
  }))
  return defs.map((x) => ({
    ...x,
    count: d[x.key],
    pct: Math.round((d[x.key] / total) * 1000) / 10,
  }))
})

onMounted(() => void load())
</script>

<template>
  <div v-loading="loading" class="dashboard">
    <div class="toolbar">
      <span class="toolbar-hint">数据来自本地审核库，操作后点击刷新</span>
      <el-button :icon="Refresh" @click="load">刷新</el-button>
    </div>

    <!-- KPI -->
    <el-row :gutter="16" class="kpi-row">
      <el-col v-for="k in kpis" :key="k.label" :xs="12" :sm="12" :md="6">
        <div class="kpi-card">
          <div class="kpi-top">
            <span class="kpi-icon" :style="{ background: k.tint, color: k.color }">{{ k.icon }}</span>
            <span class="kpi-label">{{ k.label }}</span>
          </div>
          <div class="kpi-value">{{ k.value }}</div>
          <div class="kpi-caption">{{ k.caption }}</div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="16">
      <!-- 近 7 日处理量 -->
      <el-col :xs="24" :md="14">
        <el-card shadow="never" class="panel">
          <template #header>
            <div class="panel-header">
              <span>近 7 日处理量</span>
              <div class="legend">
                <span class="legend-item"><i class="dot" :style="{ background: STATUS_COLOR.approved }" />通过</span>
                <span class="legend-item"><i class="dot" :style="{ background: STATUS_COLOR.rejected }" />驳回</span>
              </div>
            </div>
          </template>
          <div class="chart-plot">
            <div class="plot-inner">
              <div class="grid-line" style="bottom: 75%" />
              <div class="grid-line" style="bottom: 50%" />
              <div class="grid-line" style="bottom: 25%" />
              <div class="grid-line grid-line-base" />
              <div v-for="d in chart.days" :key="d.date" class="chart-group">
                <div class="bars">
                  <el-tooltip :content="`${d.date} 通过 ${d.approved} 条`" placement="top">
                    <div class="bar bar-approve" :style="{ height: barHeight(d.approved) }">
                      <span v-if="d.approved" class="bar-value">{{ d.approved }}</span>
                    </div>
                  </el-tooltip>
                  <el-tooltip :content="`${d.date} 驳回 ${d.rejected} 条`" placement="top">
                    <div class="bar bar-reject" :style="{ height: barHeight(d.rejected) }">
                      <span v-if="d.rejected" class="bar-value">{{ d.rejected }}</span>
                    </div>
                  </el-tooltip>
                </div>
                <div class="group-label">{{ d.date.slice(5) }}</div>
              </div>
            </div>
          </div>
          <el-table :data="chart.days" size="small" class="mini-table">
            <el-table-column label="日期" prop="date" width="110" />
            <el-table-column label="通过" align="center">
              <template #default="{ row }">{{ row.approved }}</template>
            </el-table-column>
            <el-table-column label="驳回" align="center">
              <template #default="{ row }">{{ row.rejected }}</template>
            </el-table-column>
            <el-table-column label="合计" align="center">
              <template #default="{ row }">{{ row.approved + row.rejected }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <!-- 状态分布 -->
      <el-col :xs="24" :md="10">
        <el-card shadow="never" class="panel">
          <template #header>
            <div class="panel-header"><span>投稿状态分布</span></div>
          </template>
          <div class="stack-bar">
            <div
              v-for="s in distribution"
              :key="s.key"
              class="stack-seg"
              :style="{ width: `${s.pct}%`, background: s.color }"
            />
          </div>
          <div class="dist-legend">
            <div v-for="s in distribution" :key="s.key" class="dist-item">
              <span class="dist-dot" :style="{ background: s.color }" />
              <span class="dist-label">{{ s.label }}</span>
              <span class="dist-count">{{ s.count }} 条 · {{ s.pct }}%</span>
            </div>
          </div>
        </el-card>

        <!-- 最近动态 -->
        <el-card shadow="never" class="panel activity-panel">
          <template #header>
            <div class="panel-header">
              <span>最近动态</span>
              <el-button link type="primary" @click="router.push('/logs')">查看全部</el-button>
            </div>
          </template>
          <div v-if="stats && stats.recentLogs.length" class="activity">
            <div v-for="l in stats.recentLogs" :key="l.id" class="activity-item">
              <el-tag :type="ACTION_TAG[l.action]" size="small" class="activity-tag">
                {{ ACTION_LABEL[l.action] }}
              </el-tag>
              <span class="activity-title">{{ l.submissionTitle }}</span>
              <span class="activity-meta">{{ l.operator }} · {{ fmtTime(l.createdAt) }}</span>
            </div>
          </div>
          <el-empty v-else description="暂无动态" :image-size="80" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.dashboard {
  max-width: 1280px;
  margin: 0 auto;
}

.toolbar {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}

.toolbar-hint {
  font-size: 12px;
  color: #a8abb2;
}

.kpi-row {
  margin-bottom: 4px;
}

.kpi-card {
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  padding: 16px 18px;
  margin-bottom: 16px;
}

.kpi-top {
  display: flex;
  align-items: center;
  gap: 8px;
}

.kpi-icon {
  width: 26px;
  height: 26px;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}

.kpi-label {
  font-size: 13px;
  color: #909399;
}

.kpi-value {
  font-size: 30px;
  font-weight: 600;
  line-height: 1.3;
  margin-top: 6px;
  color: #303133;
  font-variant-numeric: tabular-nums;
}

.kpi-caption {
  font-size: 12px;
  color: #a8abb2;
  margin-top: 2px;
}

.panel {
  margin-bottom: 16px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  font-weight: 600;
}

.legend {
  display: flex;
  gap: 14px;
  font-size: 12px;
  font-weight: 400;
  color: #606266;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.dot {
  width: 9px;
  height: 9px;
  border-radius: 2px;
  display: inline-block;
}

.chart-plot {
  padding-top: 22px;
}

.plot-inner {
  position: relative;
  height: 200px;
  display: flex;
  align-items: stretch;
  gap: 12px;
}

.grid-line {
  position: absolute;
  left: 0;
  right: 0;
  border-top: 1px dashed #e8eaee;
  z-index: 0;
}

.grid-line-base {
  bottom: 0;
  border-top: 1px solid #dcdfe6;
}

.chart-group {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
}

.bars {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: calc(100% - 24px);
  z-index: 1;
}

.bar {
  width: 22px;
  border-radius: 4px 4px 0 0;
  position: relative;
}

.bar-approve {
  background: #67c23a;
}

.bar-reject {
  background: #f56c6c;
}

.bar-value {
  position: absolute;
  top: -20px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 12px;
  color: #606266;
  font-variant-numeric: tabular-nums;
}

.group-label {
  margin-top: 6px;
  font-size: 12px;
  color: #909399;
  height: 18px;
}

.mini-table {
  margin-top: 8px;
}

.stack-bar {
  display: flex;
  gap: 2px;
  height: 28px;
  border-radius: 6px;
  overflow: hidden;
  margin: 6px 0 16px;
}

.stack-seg {
  border-radius: 2px;
  min-width: 0;
}

.dist-legend {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.dist-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.dist-dot {
  width: 10px;
  height: 10px;
  border-radius: 3px;
  flex: none;
}

.dist-label {
  color: #606266;
  min-width: 52px;
}

.dist-count {
  color: #909399;
  font-variant-numeric: tabular-nums;
}

.activity-panel {
  min-height: 200px;
}

.activity {
  display: flex;
  flex-direction: column;
}

.activity-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #f2f3f5;
}

.activity-item:last-child {
  border-bottom: none;
}

.activity-tag {
  flex: none;
}

.activity-title {
  flex: 1;
  font-size: 13px;
  color: #303133;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.activity-meta {
  flex: none;
  font-size: 12px;
  color: #a8abb2;
}
</style>
