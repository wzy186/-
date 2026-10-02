<script setup>
import { ref, reactive, onMounted } from 'vue'
import { adminRedPacketList, adminRedPacketDetail } from '../../api'

const list = ref([])
const detail = ref(null)
const dialog = reactive({ show: false })

async function load() {
  const r = await adminRedPacketList()
  list.value = r.data || []
}

async function viewDetail(id) {
  const r = await adminRedPacketDetail(id)
  detail.value = r.data
  dialog.show = true
}

function statusText(s) {
  return { 1: '进行中', 2: '已抢完', 3: '已退款关闭' }[s] || '未知'
}
function statusType(s) {
  return { 1: 'success', 2: 'info', 3: 'warning' }[s] || ''
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-title">红包雨数据</div>
    <div class="page-sub">查看所有红包雨场次及实时领取进度</div>

    <div class="card">
      <div class="card-title">
        <h3>场次列表</h3>
        <el-button size="small" @click="load">刷新</el-button>
      </div>
      <el-table :data="list" size="small">
        <el-table-column prop="id" label="场次 ID" width="180" />
        <el-table-column prop="title" label="标题" />
        <el-table-column label="总金额">
          <template #default="{ row }">¥{{ (row.totalAmount / 100).toFixed(2) }}</template>
        </el-table-column>
        <el-table-column prop="count" label="个数" width="80" />
        <el-table-column label="已领/剩余" width="110">
          <template #default="{ row }">{{ row.gotCount }} / {{ row.remainCount }}</template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="创建时间" width="170" />
        <el-table-column label="操作" width="90">
          <template #default="{ row }">
            <el-button size="small" @click="viewDetail(row.id)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-dialog v-model="dialog.show" title="场次实时详情" width="600px">
      <div v-if="detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="标题">{{ detail.info?.title }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="statusType(detail.info?.status)" size="small">{{ statusText(detail.info?.status) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="总金额">¥{{ (detail.info?.totalAmount / 100).toFixed(2) }}</el-descriptions-item>
          <el-descriptions-item label="总个数">{{ detail.info?.count }}</el-descriptions-item>
          <el-descriptions-item label="Redis 已领">{{ detail.takenCount }}</el-descriptions-item>
          <el-descriptions-item label="DB 已领">{{ detail.info?.gotCount }}</el-descriptions-item>
          <el-descriptions-item label="Redis 剩余" :span="2">{{ detail.meta?.remain }}</el-descriptions-item>
        </el-descriptions>
        <p class="muted" style="margin-top:12px">
          说明:Redis 数据实时更新,DB 数据由异步落库消费,可能有短暂延迟。
        </p>
      </div>
    </el-dialog>
  </div>
</template>
