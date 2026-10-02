import { createRouter, createWebHashHistory } from 'vue-router'
import { getToken, getUser } from '../utils/request'

const routes = [
  { path: '/login', name: 'login', component: () => import('../views/Login.vue') },
  {
    path: '/admin',
    component: () => import('../views/admin/AdminLayout.vue'),
    redirect: '/admin/dashboard',
    children: [
      { path: 'dashboard', name: 'adminDashboard', component: () => import('../views/admin/Dashboard.vue'), meta: { title: '数据概览' } },
      { path: 'voucher', name: 'adminVoucher', component: () => import('../views/admin/VoucherManage.vue'), meta: { title: '秒杀券管理' } },
      { path: 'redpacket', name: 'adminRedPacket', component: () => import('../views/admin/RedPacketData.vue'), meta: { title: '红包雨数据' } }
    ]
  },
  {
    path: '/',
    component: () => import('../layout/Layout.vue'),
    redirect: '/shop',
    children: [
      { path: 'shop', name: 'shop', component: () => import('../views/Shop.vue'), meta: { title: '商铺' } },
      { path: 'shop/:id', name: 'shopDetail', component: () => import('../views/ShopDetail.vue'), meta: { title: '商铺详情' } },
      { path: 'voucher', name: 'voucher', component: () => import('../views/Voucher.vue'), meta: { title: '优惠券秒杀' } },
      { path: 'redpacket', name: 'redpacket', component: () => import('../views/RedPacket.vue'), meta: { title: '红包雨' } },
      { path: 'blog', name: 'blog', component: () => import('../views/Blog.vue'), meta: { title: '探店笔记' } },
      { path: 'blog/:id', name: 'blogDetail', component: () => import('../views/BlogDetail.vue'), meta: { title: '笔记详情' } },
      { path: 'profile/:id', name: 'profile', component: () => import('../views/Profile.vue'), meta: { title: '个人主页' } },
      { path: 'follow', name: 'follow', component: () => import('../views/Follow.vue'), meta: { title: '关注/Feed' } },
      { path: 'stats', name: 'stats', component: () => import('../views/Stats.vue'), meta: { title: '签到/UV' } },
      { path: 'nearby', name: 'nearby', component: () => import('../views/Nearby.vue'), meta: { title: '附近商铺' } },
      { path: 'ai', name: 'ai', component: () => import('../views/Ai.vue'), meta: { title: 'AI 省钱助手' } }
    ]
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

// 路由守卫:未登录跳 /login;普通用户访问 /admin 跳回首页
router.beforeEach((to, from, next) => {
  if (to.path === '/login') {
    next()
  } else if (!getToken()) {
    next('/login')
  } else if (to.path.startsWith('/admin')) {
    const u = getUser()
    if (u && u.role === 1) next()
    else { next('/shop'); alert('仅管理员可访问后台') }
  } else {
    next()
  }
})

export default router
