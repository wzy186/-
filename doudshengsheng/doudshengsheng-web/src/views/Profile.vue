<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { queryUser, userProfile, blogOfUser, follow } from '../api'

const route = useRoute()
const router = useRouter()
const user = ref(null)
const profile = ref(null)
const blogs = ref([])

async function load() {
  const uid = route.params.id
  const [u, p, b] = await Promise.all([
    queryUser(uid),
    userProfile(uid),
    blogOfUser(uid)
  ])
  user.value = u.data
  profile.value = p.data
  blogs.value = b.data || []
}

async function toggleFollow() {
  if (!profile.value) return
  await follow(route.params.id, !profile.value.isFollowing)
  ElMessage.success(profile.value.isFollowing ? '已取消关注' : '关注成功')
  load()
}

function toBlog(id) { router.push(`/blog/${id}`) }

onMounted(load)
</script>

<template>
  <div class="page" v-if="user && profile">
    <!-- 用户信息卡 -->
    <div class="profile-card">
      <div class="pc-bg"></div>
      <div class="pc-body">
        <el-avatar :size="80" style="background:var(--brand);font-size:32px">{{ (user.nickName || 'U')[0] }}</el-avatar>
        <div class="pc-info">
          <h1>{{ user.nickName }}</h1>
          <div class="pc-stats">
            <span><b>{{ profile.following }}</b> 关注</span>
            <span><b>{{ profile.followers }}</b> 粉丝</span>
            <span><b>{{ blogs.length }}</b> 笔记</span>
          </div>
        </div>
        <el-button v-if="!profile.isMe"
          :type="profile.isFollowing ? 'default' : 'primary'" size="large" round
          @click="toggleFollow">
          {{ profile.isFollowing ? '已关注' : '+ 关注' }}
        </el-button>
        <el-tag v-else type="info">这是我自己</el-tag>
      </div>
    </div>

    <!-- TA 的笔记 -->
    <div class="card">
      <h3>TA 的笔记</h3>
      <div class="blog-list" v-if="blogs.length">
        <div v-for="b in blogs" :key="b.id" class="blog-row" @click="toBlog(b.id)">
          <div class="br-info">
            <h4>{{ b.title }}</h4>
            <span class="muted">{{ b.content?.slice(0, 40) }}...</span>
          </div>
          <div class="br-meta">
            <span>❤ {{ b.liked }}</span>
          </div>
        </div>
      </div>
      <el-empty v-else description="还没有发布笔记" />
    </div>
  </div>
  <el-empty v-else description="加载中..." />
</template>

<style scoped>
.profile-card { background:#fff; border-radius:14px; overflow:hidden; box-shadow:var(--shadow-sm); margin-bottom:16px; }
.pc-bg { height:100px; background:linear-gradient(135deg,#FF5A36,#FF8A5C); }
.pc-body { display:flex; align-items:center; gap:20px; padding:0 24px 20px; margin-top:-40px; position:relative; }
.pc-info { flex:1; }
.pc-info h1 { font-size:20px; margin-bottom:8px; }
.pc-stats { display:flex; gap:20px; font-size:13px; color:var(--ink-3); }
.pc-stats b { color:var(--ink); font-size:16px; margin-right:4px; }
.blog-list { margin-top:12px; }
.blog-row { display:flex; justify-content:space-between; align-items:center; padding:12px 0; border-bottom:1px solid var(--line); cursor:pointer; }
.blog-row:hover { background:var(--brand-light); }
.br-info h4 { font-size:15px; margin-bottom:4px; }
.br-meta { color:var(--brand); font-size:14px; }
</style>
