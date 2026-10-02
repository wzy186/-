<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { adminAddVoucher, adminVoucherStock, adminPreheat } from '../../api'

const form = reactive({
  shopId: 7,
  title: '9.9元秒杀电影票',
  subTitle: '限量抢购',
  payValue: 990,
  actualValue: 4500,
  type: 2,
  status: 1
})
const stock = ref(null)
const queryId = ref(10)

async function create() {
  if (!form.title) { ElMessage.warning('请填标题'); return }
  // 金额转分
  const data = { ...form, payValue: form.payValue, actualValue: form.actualValue }
  const r = await adminAddVoucher(data)
  ElMessage.success(`创建成功,券 ID: ${r.data}`)
  queryId.value = r.data
  loadStock()
}

async function loadStock() {
  const r = await adminVoucherStock(queryId.value)
  stock.value = r.data
}

async function preheat() {
  await adminPreheat(queryId.value)
  ElMessage.success('库存已重新预热')
  loadStock()
}

onMounted(loadStock)
</script>

<template>
  <div class="page">
    <div class="page-title">秒杀券管理</div>
    <div class="page-sub">创建秒杀券(自动预热库存到 Redis)、查看实时库存</div>

    <div class="card">
      <h3>创建秒杀券</h3>
      <el-form :model="form" label-width="100px" style="margin-top:12px; max-width:560px">
        <el-form-item label="商铺 ID"><el-input-number v-model="form.shopId" :min="1" /></el-form-item>
        <el-form-item label="标题"><el-input v-model="form.title" /></el-form-item>
        <el-form-item label="副标题"><el-input v-model="form.subTitle" /></el-form-item>
        <el-form-item label="支付价(分)"><el-input-number v-model="form.payValue" :min="0" /></el-form-item>
        <el-form-item label="抵扣价(分)"><el-input-number v-model="form.actualValue" :min="0" /></el-form-item>
        <el-form-item>
          <el-button type="primary" @click="create">创建并预热</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="card">
      <div class="card-title">
        <h3>库存查询</h3>
        <div class="row">
          <el-input-number v-model="queryId" :min="1" />
          <el-button @click="loadStock">查询</el-button>
          <el-button type="warning" @click="preheat">重新预热</el-button>
        </div>
      </div>
      <div v-if="stock" class="row" style="margin-top:12px">
        <el-statistic title="Redis 实时库存" :value="stock.redisStock" />
        <el-statistic title="DB 初始库存" :value="stock.dbStock" />
      </div>
    </div>
  </div>
</template>
