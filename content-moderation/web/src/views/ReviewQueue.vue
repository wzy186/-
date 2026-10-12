<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { TableInstance } from 'element-plus'
import { Refresh, Search } from '@element-plus/icons-vue'
import { api } from '../api'
import { currentOperator, loadRejectReasons, rejectReasons } from '../state'
import { CUSTOM_REASON_LABEL, STATUS_LABEL, STATUS_TAG, fmtTime } from '../constants'
import type { SubmissionItem, SubmissionStatus } from '../types'

// ---------- 列表查询 ----------
const statusTab = ref<SubmissionStatus | 'all'>('pending')
const keyword = ref('')
const sort = ref<'newest' | 'oldest'>('newest')
const page = ref(1)
const pageSize = ref(20)
const list = ref<SubmissionItem[]>([])
const total = ref(0)
const loading = ref(false)

// 请求序号：并发加载时只认最后一次请求的响应，避免旧响应覆盖新数据
let loadSeq = 0
async function load() {
  const seq = ++loadSeq
  loading.value = true
  try {
    const r = await api.submissions({
      status: statusTab.value,
      q: keyword.value || undefined,
      sort: sort.value,
      page: page.value,
      pageSize: pageSize.value,
    })
    if (seq !== loadSeq) return
    list.value = r.items
    total.value = r.total
    // 当前页被审空后自动回退到最后一个有效页（改 page 由下方 watcher 重新加载）
    const lastPage = Math.max(1, Math.ceil(r.total / pageSize.value))
    if (r.items.length === 0 && page.value > lastPage) {
      page.value = lastPage
      return
    }
  } catch (e) {
    if (seq !== loadSeq) return
    ElMessage.error(e instanceof Error ? e.message : '加载失败')
  } finally {
    if (seq === loadSeq) loading.value = false
  }
}

let searchTimer: ReturnType<typeof setTimeout> | undefined
function onKeywordInput() {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => {
    page.value = 1
    void load()
  }, 400)
}
function onKeywordCommit() {
  clearTimeout(searchTimer)
  page.value = 1
  void load()
}
onUnmounted(() => clearTimeout(searchTimer))

watch(statusTab, () => {
  page.value = 1
  void load()
})
watch([sort, pageSize], () => {
  page.value = 1
  void load()
})
watch(page, () => {
  void load()
})

// ---------- tab 角标（状态数量） ----------
const statusCounts = reactive<Record<SubmissionStatus | 'all', number>>({
  pending: 0,
  approved: 0,
  rejected: 0,
  all: 0,
})
async function refreshStats() {
  try {
    const s = await api.stats()
    statusCounts.pending = s.statusDistribution.pending
    statusCounts.approved = s.statusDistribution.approved
    statusCounts.rejected = s.statusDistribution.rejected
    statusCounts.all = s.pendingCount + s.statusDistribution.approved + s.statusDistribution.rejected
  } catch {
    // 角标刷新失败不影响主流程
  }
}

// ---------- 勾选 / 批量 ----------
const tableRef = ref<TableInstance>()
const selectedRows = ref<SubmissionItem[]>([])
function onSelectionChange(rows: SubmissionItem[]) {
  selectedRows.value = rows
}
function clearSelection() {
  tableRef.value?.clearSelection()
}

function requireOperator(): string {
  if (!currentOperator.value) {
    ElMessage.warning('请先在右上角选择「当前操作人」')
    return ''
  }
  return currentOperator.value
}

// 审核动作进行中：防止双击重复提交
const actionBusy = ref(false)

// ---------- 单个操作 ----------
async function doReview(row: SubmissionItem, action: 'approve' | 'reopen') {
  const operator = requireOperator()
  if (!operator) return
  actionBusy.value = true
  try {
    await api.review(row.id, { action, operator })
    ElMessage.success(action === 'approve' ? `已通过「${row.title}」` : `「${row.title}」已重新进入待审核`)
    clearSelection()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : '操作失败')
  } finally {
    actionBusy.value = false
  }
  // 无论成败都刷新：失败多因条目已被他人处理（409），列表里需要同步最新状态
  await Promise.all([load(), refreshStats()])
  if (detailVisible.value) await refreshDetail()
}

async function onReopen(row: SubmissionItem) {
  try {
    await ElMessageBox.confirm(`确认将「${row.title}」重新放入待审核队列？`, '重新审核', {
      type: 'warning',
      confirmButtonText: '确认',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  await doReview(row, 'reopen')
}

async function onBatchApprove() {
  const n = selectedRows.value.length
  if (!n) return
  const operator = requireOperator()
  if (!operator) return
  try {
    await ElMessageBox.confirm(`确认通过选中的 ${n} 条投稿？`, '批量通过', {
      type: 'warning',
      confirmButtonText: '确认通过',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  actionBusy.value = true
  try {
    const r = await api.batchReview({ ids: selectedRows.value.map((x) => x.id), action: 'approve', operator })
    let msg = `已通过 ${r.updated} 条`
    if (r.skipped > 0) msg += `，跳过 ${r.skipped} 条（已不在待审核状态）`
    if (r.updated === 0) ElMessage.warning(msg)
    else ElMessage.success(msg)
    clearSelection()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : '操作失败')
  } finally {
    actionBusy.value = false
  }
  await Promise.all([load(), refreshStats()])
}

// ---------- 驳回弹窗（单个 / 批量共用） ----------
interface RejectOption {
  label: string
  custom: boolean
}
const rejectVisible = ref(false)
const rejectMode = ref<'single' | 'batch'>('single')
const rejectTarget = ref<SubmissionItem | null>(null)
const rejectChoice = ref<RejectOption>({ label: '', custom: false })
const rejectNote = ref('')
const rejectSubmitting = ref(false)

const rejectOptions = computed<RejectOption[]>(() => [
  ...rejectReasons.value.map((label) => ({ label, custom: false })),
  { label: CUSTOM_REASON_LABEL, custom: true },
])
const rejectTitle = computed(() =>
  rejectMode.value === 'batch' ? `批量驳回 ${selectedRows.value.length} 条投稿` : '驳回投稿',
)
const needCustomReason = computed(() => rejectChoice.value.custom)

function resetRejectForm() {
  rejectChoice.value = rejectOptions.value[0] ?? { label: '', custom: true }
  rejectNote.value = ''
}
function openReject(row: SubmissionItem) {
  rejectMode.value = 'single'
  rejectTarget.value = row
  resetRejectForm()
  rejectVisible.value = true
}
function openBatchReject() {
  if (!selectedRows.value.length) return
  rejectMode.value = 'batch'
  rejectTarget.value = null
  resetRejectForm()
  rejectVisible.value = true
}

function composeReason(): string | null {
  if (rejectChoice.value.custom) return rejectNote.value.trim() || null
  const note = rejectNote.value.trim()
  return note ? `${rejectChoice.value.label} · ${note}` : rejectChoice.value.label
}

async function submitReject() {
  const reason = composeReason()
  if (!reason) {
    ElMessage.warning('请填写具体驳回原因')
    return
  }
  // 弹窗开着时用户可能取消了勾选，提交前复核
  if (rejectMode.value === 'batch' && !selectedRows.value.length) {
    rejectVisible.value = false
    ElMessage.warning('未选择任何投稿')
    return
  }
  const operator = requireOperator()
  if (!operator) return
  rejectSubmitting.value = true
  try {
    if (rejectMode.value === 'batch') {
      const r = await api.batchReview({ ids: selectedRows.value.map((x) => x.id), action: 'reject', reason, operator })
      let msg = `已驳回 ${r.updated} 条`
      if (r.skipped > 0) msg += `，跳过 ${r.skipped} 条（已不在待审核状态）`
      if (r.updated === 0) ElMessage.warning(msg)
      else ElMessage.success(msg)
    } else if (rejectTarget.value) {
      await api.review(rejectTarget.value.id, { action: 'reject', reason, operator })
      ElMessage.success(`已驳回「${rejectTarget.value.title}」`)
    }
    rejectVisible.value = false
    clearSelection()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : '操作失败')
  } finally {
    rejectSubmitting.value = false
  }
  await Promise.all([load(), refreshStats()])
  if (detailVisible.value) await refreshDetail()
}

// ---------- 详情弹窗 ----------
const detailVisible = ref(false)
const detailLoading = ref(false)
const detailItem = ref<SubmissionItem | null>(null)

async function openDetail(row: SubmissionItem) {
  detailVisible.value = true
  detailItem.value = null
  detailLoading.value = true
  try {
    detailItem.value = await api.submission(row.id)
  } catch (e) {
    // 取不到详情（如条目已被删除）时直接关掉弹窗，避免空壳对话框
    detailVisible.value = false
    ElMessage.error(e instanceof Error ? e.message : '加载详情失败')
  } finally {
    detailLoading.value = false
  }
}
async function refreshDetail() {
  if (!detailItem.value) return
  try {
    detailItem.value = await api.submission(detailItem.value.id)
  } catch {
    // 忽略刷新失败
  }
}
function detailApprove() {
  if (detailItem.value) void doReview(detailItem.value, 'approve')
}
function detailReject() {
  if (detailItem.value) openReject(detailItem.value)
}
function detailReopen() {
  if (detailItem.value) void onReopen(detailItem.value)
}

onMounted(() => {
  void load()
  void refreshStats()
  void loadRejectReasons()
})
</script>

<template>
  <el-card shadow="never" class="page-card">
    <el-tabs v-model="statusTab" class="status-tabs">
      <el-tab-pane name="pending">
        <template #label>
          <span>待审核<el-badge v-if="statusCounts.pending" :value="statusCounts.pending" :max="999" class="tab-badge" /></span>
        </template>
      </el-tab-pane>
      <el-tab-pane name="approved">
        <template #label>
          <span>已通过<el-badge v-if="statusCounts.approved" :value="statusCounts.approved" :max="999" class="tab-badge" /></span>
        </template>
      </el-tab-pane>
      <el-tab-pane name="rejected">
        <template #label>
          <span>已驳回<el-badge v-if="statusCounts.rejected" :value="statusCounts.rejected" :max="999" class="tab-badge" /></span>
        </template>
      </el-tab-pane>
      <el-tab-pane name="all">
        <template #label>
          <span>全部<el-badge v-if="statusCounts.all" :value="statusCounts.all" :max="999" class="tab-badge" /></span>
        </template>
      </el-tab-pane>
    </el-tabs>

    <div class="toolbar">
      <el-input
        v-model="keyword"
        placeholder="搜索标题 / 内容 / 作者"
        clearable
        class="search"
        @input="onKeywordInput"
        @keyup.enter="onKeywordCommit"
        @clear="onKeywordCommit"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
      </el-input>
      <el-select v-model="sort" style="width: 130px">
        <el-option label="最新提交" value="newest" />
        <el-option label="最早提交" value="oldest" />
      </el-select>
      <el-button :icon="Refresh" @click="load">刷新</el-button>
    </div>

    <div v-if="selectedRows.length" class="batch-bar">
      <span class="batch-count">已选 <b>{{ selectedRows.length }}</b> 条</span>
      <el-button type="success" size="small" :disabled="actionBusy" @click="onBatchApprove">批量通过</el-button>
      <el-button type="danger" size="small" :disabled="actionBusy" @click="openBatchReject">批量驳回</el-button>
      <el-button link size="small" @click="clearSelection">取消选择</el-button>
    </div>

    <el-table
      ref="tableRef"
      :data="list"
      v-loading="loading"
      row-key="id"
      class="queue-table"
      @selection-change="onSelectionChange"
    >
      <el-table-column type="selection" width="44" :reserve-selection="true" />
      <el-table-column label="封面" width="96">
        <template #default="{ row }">
          <el-image
            v-if="row.imageUrls.length"
            :src="row.imageUrls[0]"
            :preview-src-list="row.imageUrls"
            preview-teleported
            fit="cover"
            lazy
            class="cover"
          />
          <div v-else class="cover cover-empty">无图</div>
        </template>
      </el-table-column>
      <el-table-column label="标题 / 摘要" min-width="300">
        <template #default="{ row }">
          <div class="cell-title">
            <el-button link type="primary" class="title-link" @click="openDetail(row)">{{ row.title }}</el-button>
          </div>
          <div class="cell-excerpt">{{ row.excerpt }}</div>
        </template>
      </el-table-column>
      <el-table-column prop="author" label="作者" width="130" />
      <el-table-column label="状态" width="90">
        <template #default="{ row }">
          <el-tag :type="STATUS_TAG[row.status as SubmissionStatus]" size="small">
            {{ STATUS_LABEL[row.status as SubmissionStatus] }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="提交时间" width="150">
        <template #default="{ row }">{{ fmtTime(row.createdAt, true) }}</template>
      </el-table-column>
      <el-table-column label="审核信息" min-width="180">
        <template #default="{ row }">
          <span v-if="row.status === 'pending'" class="text-muted">—</span>
          <div v-else-if="row.status === 'rejected'" class="review-info">
            <div>{{ row.reviewedBy ?? '—' }} · {{ fmtTime(row.reviewedAt, true) }}</div>
            <el-tooltip :content="row.rejectReason ?? ''" placement="top" :show-after="300">
              <div class="reject-reason">{{ row.rejectReason }}</div>
            </el-tooltip>
          </div>
          <div v-else>{{ row.reviewedBy ?? '—' }} · {{ fmtTime(row.reviewedAt, true) }}</div>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="190" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" size="small" @click="openDetail(row)">详情</el-button>
          <template v-if="row.status === 'pending'">
            <el-button link type="success" size="small" :disabled="actionBusy" @click="doReview(row, 'approve')">通过</el-button>
            <el-button link type="danger" size="small" :disabled="actionBusy" @click="openReject(row)">驳回</el-button>
          </template>
          <el-button v-else link type="warning" size="small" :disabled="actionBusy" @click="onReopen(row)">重新审核</el-button>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty
          :description="statusTab === 'pending' ? '待审核队列已清空，干得漂亮 🎉' : '暂无数据'"
          :image-size="90"
        />
      </template>
    </el-table>

    <div class="pager">
      <el-pagination
        v-model:current-page="page"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next, jumper"
        background
      />
    </div>
  </el-card>

  <!-- 详情弹窗 -->
  <el-dialog v-model="detailVisible" width="640px">
    <template #header>
      <span class="dialog-title">投稿详情 #{{ detailItem?.id ?? '' }}</span>
    </template>
    <div v-loading="detailLoading" class="detail-body">
      <template v-if="detailItem">
        <div class="detail-meta">
          <el-tag :type="STATUS_TAG[detailItem.status]" size="small">
            {{ STATUS_LABEL[detailItem.status] }}
          </el-tag>
          <span>{{ detailItem.author }}</span>
          <span>{{ fmtTime(detailItem.createdAt, true) }}</span>
        </div>
        <h3 class="detail-title">{{ detailItem.title }}</h3>
        <p class="detail-content">{{ detailItem.content }}</p>
        <div v-if="detailItem.imageUrls.length" class="detail-images">
          <el-image
            v-for="(u, i) in detailItem.imageUrls"
            :key="i"
            :src="u"
            :preview-src-list="detailItem.imageUrls"
            :initial-index="i"
            preview-teleported
            fit="cover"
            lazy
            class="detail-img"
          />
        </div>
        <div v-if="detailItem.status !== 'pending'" class="detail-review">
          <div class="detail-review-title">审核记录</div>
          <div>
            {{ STATUS_LABEL[detailItem.status] }} · {{ detailItem.reviewedBy ?? '—' }} ·
            {{ fmtTime(detailItem.reviewedAt, true) }}
          </div>
          <div v-if="detailItem.rejectReason" class="reject-reason">原因：{{ detailItem.rejectReason }}</div>
        </div>
      </template>
    </div>
    <template #footer>
      <template v-if="detailItem">
        <el-button @click="detailVisible = false">关闭</el-button>
        <template v-if="detailItem.status === 'pending'">
          <el-button type="success" :disabled="actionBusy" @click="detailApprove">通过</el-button>
          <el-button type="danger" :disabled="actionBusy" @click="detailReject">驳回</el-button>
        </template>
        <el-button v-else type="warning" :disabled="actionBusy" @click="detailReopen">重新审核</el-button>
      </template>
    </template>
  </el-dialog>

  <!-- 驳回弹窗 -->
  <el-dialog v-model="rejectVisible" :title="rejectTitle" width="480px">
    <div class="reject-form">
      <div class="reject-label">驳回原因</div>
      <el-radio-group v-model="rejectChoice">
        <el-radio v-for="o in rejectOptions" :key="o.label" :value="o">{{ o.label }}</el-radio>
      </el-radio-group>
      <el-input
        v-model="rejectNote"
        type="textarea"
        :rows="3"
        :placeholder="needCustomReason ? '请填写具体原因' : '补充说明（可选，会一并记录）'"
        class="reject-note"
      />
    </div>
    <template #footer>
      <el-button @click="rejectVisible = false">取消</el-button>
      <el-button type="danger" :loading="rejectSubmitting" @click="submitReject">确认驳回</el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.page-card {
  max-width: 1280px;
  margin: 0 auto;
}

.status-tabs :deep(.el-tabs__header) {
  margin-bottom: 14px;
}

.tab-badge {
  margin-left: 8px;
}

.toolbar {
  display: flex;
  gap: 12px;
  margin-bottom: 14px;
}

.search {
  width: 300px;
}

.batch-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #ecf5ff;
  border: 1px solid #d9ecff;
  border-radius: 4px;
  padding: 8px 14px;
  margin-bottom: 12px;
}

.batch-count {
  font-size: 14px;
  color: #409eff;
  margin-right: 4px;
}

.cover {
  width: 72px;
  height: 48px;
  border-radius: 4px;
  display: block;
}

.cover-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f4f4f5;
  color: #c0c4cc;
  font-size: 12px;
}

.cell-title {
  margin-bottom: 2px;
}

.title-link {
  padding: 0;
  font-size: 14px;
  justify-content: flex-start;
}

.cell-excerpt {
  font-size: 12px;
  color: #909399;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.text-muted {
  color: #c0c4cc;
}

.review-info {
  font-size: 12px;
  color: #909399;
}

.reject-reason {
  font-size: 12px;
  color: #f56c6c;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
}

.pager {
  display: flex;
  justify-content: flex-end;
  margin-top: 14px;
}

.dialog-title {
  font-weight: 600;
}

.detail-body {
  min-height: 120px;
}

.detail-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: #909399;
  margin-bottom: 10px;
}

.detail-title {
  margin: 0 0 10px;
  font-size: 17px;
}

.detail-content {
  margin: 0;
  font-size: 14px;
  line-height: 1.7;
  color: #303133;
  white-space: pre-wrap;
}

.detail-images {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}

.detail-img {
  width: 180px;
  height: 120px;
  border-radius: 4px;
}

.detail-review {
  margin-top: 16px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 4px;
  font-size: 13px;
  color: #606266;
  line-height: 1.8;
}

.detail-review-title {
  font-weight: 600;
  color: #303133;
}

.reject-form {
  padding: 4px 0;
}

.reject-label {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
}

.reject-form :deep(.el-radio) {
  margin-right: 20px;
}

.reject-note {
  margin-top: 14px;
}
</style>
