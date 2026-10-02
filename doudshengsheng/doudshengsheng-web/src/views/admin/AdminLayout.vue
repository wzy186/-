<script setup>
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

const menus = [
  { index: '/admin/dashboard', label: '数据概览', icon: 'DataAnalysis' },
  { index: '/admin/voucher', label: '秒杀券管理', icon: 'Ticket' },
  { index: '/admin/redpacket', label: '红包雨数据', icon: 'Present' }
]
</script>

<template>
  <el-container style="height:100%">
    <el-header class="admin-top">
      <div class="row">
        <span class="brand">兜省省 · 商户后台</span>
      </div>
      <div class="row">
        <el-button text size="small" @click="router.push('/shop')">返回用户端 →</el-button>
      </div>
    </el-header>
    <el-container>
      <el-aside width="200px" class="admin-side">
        <el-menu :default-active="route.path" router class="side-menu">
          <el-menu-item v-for="m in menus" :key="m.index" :index="m.index">
            <el-icon><component :is="m.icon" /></el-icon>
            <span>{{ m.label }}</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main class="admin-main">
        <RouterView />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.admin-top { background: #1f2937; color: #fff; display:flex; align-items:center; justify-content:space-between; height:56px !important; padding:0 24px; }
.brand { font-size:16px; font-weight:600; }
.admin-side { background:#fff; border-right:1px solid var(--line); }
.side-menu { border-right:none !important; }
.admin-main { background:#f5f7fa; padding:0; }
</style>
