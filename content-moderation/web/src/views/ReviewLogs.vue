<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { api } from '../api'
import { ACTION_LABEL, ACTION_TAG, fmtTime } from '../constants'
import type { Operator, ReviewAction, ReviewLog } from '../types'

const operators = ref<Operator[]>([])
const operatorFilter = ref('')
const actionFilter = ref<ReviewAction | ''>('')
const page = ref(1)
const pageSize = ref(20)
const list = ref<ReviewLog[]>([])
const total = ref(0)
const loading = ref(false)

// 请求序号：并发加载时只认最后一次请求的响应
let loadSeq = 0
async function load() {
  const seq = ++loadSeq
  loading.value = true
  try {
    const r = await api.logs({
      operator: operatorFilter.value || undefined,
      action: actionFilter.value || undefined,
      page: page.value,
      pageSize: pageSize.value,
    })
    if (seq !== loadSeq) return
    list.value = r.items
    total.value = r.total
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

watch([operatorFilter, actionFilter], () => {
  page.value = 1
  void load()
})
watch(page, () => {
  void load()
})
// 切换每页条数时：若 page 已是 1 不会触发 page watcher，需单独兜底
watch(pageSize, () => {
  page.value = 1
  void load()
})

onMounted(() => {
  // 操作人下拉与日志列表并行加载，互不阻塞
  void load()
  api
    .operators()
    .then((o) => {
      operators.value = o
    })
    .catch(() => {
      // 操作人下拉加载失败不阻塞日志
    })
})
</script>

<template>
  <el-card shadow="never" class="page-card">
    <div class="toolbar">
      <el-select v-model="operatorFilter" placeholder="全部操作人" clearable style="width: 150px">
        <el-option v-for="op in operators" :key="op.id" :label="op.name" :value="op.name" />
      </el-select>
      <el-select v-model="actionFilter" placeholder="全部动作" clearable style="width: 150px">
        <el-option label="通过" value="approve" />
        <el-option label="驳回" value="reject" />
        <el-option label="重新审核" value="reopen" />
      </el-select>
      <el-button :icon="Refresh" @click="load">刷新</el-button>
    </div>

    <el-table :data="list" v-loading="loading" class="logs-table">
      <el-table-column label="时间" width="160">
        <template #default="{ row }">{{ fmtTime(row.createdAt, true) }}</template>
      </el-table-column>
      <el-table-column prop="operator" label="操作人" width="110" />
      <el-table-column label="动作" width="110">
        <template #default="{ row }">
          <el-tag :type="ACTION_TAG[row.action as ReviewAction]" size="small">
            {{ ACTION_LABEL[row.action as ReviewAction] }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="submissionTitle" label="投稿" min-width="300" show-overflow-tooltip />
      <el-table-column label="原因 / 备注" min-width="200">
        <template #default="{ row }">
          <span v-if="row.reason">{{ row.reason }}</span>
          <span v-else class="text-muted">—</span>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="暂无日志" :image-size="90" />
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
</template>

<style scoped>
.page-card {
  max-width: 1280px;
  margin: 0 auto;
}

.toolbar {
  display: flex;
  gap: 12px;
  margin-bottom: 14px;
}

.text-muted {
  color: #c0c4cc;
}

.pager {
  display: flex;
  justify-content: flex-end;
  margin-top: 14px;
}
</style>
