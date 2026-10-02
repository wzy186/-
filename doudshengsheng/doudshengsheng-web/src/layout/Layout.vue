<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { getUser, clearToken, isAdmin } from '../utils/request'
import { me } from '../api'

const router = useRouter()
const route = useRoute()
const user = ref(getUser())
const drawerOpen = ref(false)
const isMobile = ref(window.innerWidth < 900)

const menus = [
  { index: '/shop', label: '优惠商铺', icon: 'Shop' },
  { index: '/nearby', label: '附近优惠', icon: 'Location' },
  { index: '/voucher', label: '限时秒杀', icon: 'Ticket' },
  { index: '/redpacket', label: '红包雨', icon: 'Present' },
  { index: '/blog', label: '探店笔记', icon: 'EditPen' },
  { index: '/follow', label: '我的关注', icon: 'Connection' },
  { index: '/stats', label: '签到中心', icon: 'Calendar' },
  { index: '/ai', label: 'AI 省钱助手', icon: 'ChatDotRound' }
]

onMounted(async () => {
  window.addEventListener('resize', onResize)
  try {
    const r = await me()
    if (r.data) {
      user.value = r.data
      localStorage.setItem('dss_user', JSON.stringify(r.data))
    }
  } catch (e) {}
})

onUnmounted(() => window.removeEventListener('resize', onResize))

function onResize() {
  isMobile.value = window.innerWidth < 900
  if (!isMobile.value) drawerOpen.value = false
}

function logout() {
  clearToken()
  router.push('/login')
}

function go(path) {
  router.push(path)
  drawerOpen.value = false
}
</script>

<template>
  <el-container style="height: 100%">
    <el-header class="topbar">
      <div class="topbar-left">
        <el-icon v-if="isMobile" class="burger" @click="drawerOpen = true"><Menu /></el-icon>
        <div class="brand-logo">兜省省</div>
        <span class="brand-tag" v-if="!isMobile">本地优惠</span>
      </div>
      <div class="topbar-right">
        <template v-if="user">
          <el-button v-if="isAdmin()" text size="small" @click="router.push('/admin/dashboard')">商户后台</el-button>
          <span class="user-name" v-if="!isMobile">{{ user.nickName }}</span>
          <el-avatar :size="30" style="background: var(--brand)">{{ (user.nickName || 'U')[0] }}</el-avatar>
          <el-button text @click="logout">退出</el-button>
        </template>
      </div>
    </el-header>
    <el-container>
      <!-- 桌面侧栏 -->
      <el-aside v-if="!isMobile" width="200px" class="sidebar">
        <el-menu :default-active="route.path" router class="side-menu">
          <el-menu-item v-for="m in menus" :key="m.index" :index="m.index">
            <el-icon><component :is="m.icon" /></el-icon>
            <span>{{ m.label }}</span>
          </el-menu-item>
        </el-menu>
        <div class="sidebar-foot">兜省省 · v1.0</div>
      </el-aside>

      <!-- 移动端抽屉 -->
      <el-drawer v-model="drawerOpen" direction="ltr" size="240px" :with-header="false">
        <div class="drawer-brand">兜省省</div>
        <el-menu :default-active="route.path" class="drawer-menu">
          <el-menu-item v-for="m in menus" :key="m.index" :index="m.index" @click="go(m.index)">
            <el-icon><component :is="m.icon" /></el-icon>
            <span>{{ m.label }}</span>
          </el-menu-item>
        </el-menu>
      </el-drawer>

      <el-main class="main">
        <RouterView />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.topbar {
  background: #fff; border-bottom: 1px solid var(--line);
  display: flex; align-items: center; justify-content: space-between;
  height: 56px !important; padding: 0 16px; box-shadow: var(--shadow-sm);
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.burger { font-size: 20px; cursor: pointer; color: var(--ink-2); }
.brand-logo { font-size: 19px; font-weight: 700; color: var(--brand); letter-spacing: 1px; }
.brand-tag { font-size: 11px; color: var(--ink-3); border: 1px solid var(--line); padding: 2px 8px; border-radius: 10px; }
.topbar-right { display: flex; align-items: center; gap: 10px; }
.user-name { font-size: 13px; color: var(--ink-2); }

.sidebar { background: #fff; border-right: 1px solid var(--line); display: flex; flex-direction: column; }
.side-menu { border-right: none !important; flex: 1; }
.sidebar-foot { padding: 16px; color: var(--ink-3); font-size: 11px; text-align: center; border-top: 1px solid var(--line); }
.main { background: var(--bg); padding: 0; }

.drawer-brand { padding: 20px 24px; font-size: 20px; font-weight: 700; color: var(--brand); border-bottom: 1px solid var(--line); }
.drawer-menu { border-right: none !important; }
</style>
