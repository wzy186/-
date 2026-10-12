<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { api } from './api'
import type { Operator } from './types'
import { currentOperator, setOperator } from './state'

const route = useRoute()
const operators = ref<Operator[]>([])

onMounted(async () => {
  try {
    operators.value = await api.operators()
    if (!operators.value.some((o) => o.name === currentOperator.value)) {
      setOperator(operators.value[0]?.name ?? '')
    }
  } catch (e) {
    console.error('加载操作人列表失败', e)
  }
})
</script>

<template>
  <el-container class="layout">
    <el-header class="header" height="56px">
      <div class="brand">
        <span class="brand-icon">🛡️</span>
        <span class="brand-name">投稿审核台</span>
      </div>
      <el-menu mode="horizontal" router :default-active="route.path" :ellipsis="false" class="nav">
        <el-menu-item index="/">审核队列</el-menu-item>
        <el-menu-item index="/logs">操作日志</el-menu-item>
        <el-menu-item index="/stats">统计看板</el-menu-item>
      </el-menu>
      <div class="operator">
        <span class="operator-label">当前操作人</span>
        <el-select v-model="currentOperator" style="width: 120px" @change="setOperator">
          <el-option v-for="op in operators" :key="op.id" :label="op.name" :value="op.name" />
        </el-select>
      </div>
    </el-header>
    <el-main class="main">
      <router-view />
    </el-main>
  </el-container>
</template>

<style scoped>
.layout {
  min-height: 100vh;
}

.header {
  display: flex;
  align-items: center;
  gap: 24px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  padding: 0 20px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.brand-icon {
  font-size: 20px;
}

.brand-name {
  font-size: 16px;
  font-weight: 600;
}

.nav {
  flex: 1;
  border-bottom: none;
}

.operator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.operator-label {
  font-size: 13px;
  color: #909399;
  white-space: nowrap;
}

.main {
  padding: 20px;
}
</style>
