<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { blogDetail, likeBlog, blogLikes } from '../api'

const route = useRoute()
const router = useRouter()
const blog = ref(null)
const liked = ref(false)
const likeBoard = ref(null)

const coverColors = [
  'linear-gradient(135deg,#FF9A8B,#FF6A88)',
  'linear-gradient(135deg,#667eea,#764ba2)',
  'linear-gradient(135deg,#4facfe,#00f2fe)',
  'linear-gradient(135deg,#fa709a,#fee140)',
]
function cover(id) { return coverColors[id % coverColors.length] }

async function load() {
  const r = await blogDetail(route.params.id)
  blog.value = r.data
  liked.value = r.data.isLike
}

async function toggleLike() {
  if (!blog.value) return
  await likeBlog(blog.value.id)
  liked.value = !liked.value
  ElMessage.success(liked.value ? '已点赞' : '已取消')
  load()
}

async function showBoard() {
  const r = await blogLikes(blog.value.id)
  likeBoard.value = r.data || []
}

function toShop(id) {
  if (id) router.push(`/shop/${id}`)
}
function toAuthor(uid) {
  if (uid) router.push(`/profile/${uid}`)
}

onMounted(load)
</script>

<template>
  <div class="page" v-if="blog">
    <div class="cover" :style="{ background: cover(blog.id) }">
      <span class="shop-tag" v-if="blog.shopId" @click="toShop(blog.shopId)">来自商铺 #{{ blog.shopId }} →</span>
    </div>

    <div class="card">
      <h1>{{ blog.title }}</h1>
      <div class="meta">
        <div class="author" @click="toAuthor(blog.userId)">
          <el-avatar :size="32" style="background:var(--brand)">{{ (blog.author?.nickName || 'U')[0] }}</el-avatar>
          <span>{{ blog.author?.nickName || '匿名用户' }}</span>
        </div>
        <span class="muted">{{ blog.createTime }}</span>
      </div>

      <div class="content">{{ blog.content }}</div>

      <div class="actions">
        <div :class="['like-btn', { liked }]" @click="toggleLike">
          <span class="heart">{{ liked ? '❤️' : '🤍' }}</span>
          <span>{{ blog.liked }}</span>
        </div>
        <el-button @click="showBoard">查看点赞榜</el-button>
      </div>

      <div v-if="likeBoard" class="like-board">
        <h3>点赞榜 Top{{ likeBoard.length }}</h3>
        <div class="lb-list">
          <span v-for="(u, i) in likeBoard" :key="i" class="lb-item" @click="toAuthor(u.userId)">
            {{ ['🥇','🥈','🥉'][i] || '#' + (i+1) }} {{ u.userId }}
          </span>
        </div>
      </div>
    </div>
  </div>
  <el-empty v-else description="加载中..." />
</template>

<style scoped>
.cover { height: 200px; border-radius: 14px; margin-bottom: 16px; position: relative; }
.shop-tag { position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,0.4); color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 12px; cursor: pointer; }
.card h1 { font-size: 22px; margin-bottom: 12px; }
.meta { display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid var(--line); }
.author { display: flex; align-items: center; gap: 8px; cursor: pointer; }
.author:hover { color: var(--brand); }
.content { margin: 16px 0; line-height: 1.8; font-size: 15px; white-space: pre-wrap; }
.actions { display: flex; gap: 16px; align-items: center; padding-top: 12px; border-top: 1px solid var(--line); }
.like-btn { display: flex; align-items: center; gap: 4px; cursor: pointer; padding: 6px 14px; border-radius: 16px; background: var(--bg); }
.like-btn.liked { color: var(--brand); }
.heart { font-size: 18px; }
.like-board { margin-top: 16px; }
.lb-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.lb-item { background: var(--bg); padding: 4px 10px; border-radius: 12px; font-size: 12px; cursor: pointer; }
.lb-item:hover { background: var(--brand-light); color: var(--brand); }
</style>
