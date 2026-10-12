import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'queue', component: () => import('./views/ReviewQueue.vue'), meta: { title: '审核队列' } },
    { path: '/logs', name: 'logs', component: () => import('./views/ReviewLogs.vue'), meta: { title: '操作日志' } },
    { path: '/stats', name: 'stats', component: () => import('./views/StatsDashboard.vue'), meta: { title: '统计看板' } },
  ],
})

router.afterEach((to) => {
  const title = (to.meta.title as string | undefined) ?? ''
  document.title = title ? `${title} · 投稿审核台` : '投稿审核台'
})

export default router
