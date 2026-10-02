<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { hotBlog, likeBlog, blogLikes, saveBlog } from '../api'

const router = useRouter()
const blogs = ref([])
const likedSet = ref(new Set()) // 本地点赞态
const likesMap = ref({})
const dialog = reactive({ show: false, title: '', content: '', shopId: 1 })

const coverColors = [
  'linear-gradient(135deg,#FF9A8B,#FF6A88)',
  'linear-gradient(135deg,#a8edea,#fed6e3)',
  'linear-gradient(135deg,#667eea,#764ba2)',
  'linear-gradient(135deg,#f093fb,#f5576c)',
  'linear-gradient(135deg,#4facfe,#00f2fe)',
  'linear-gradient(135deg,#fa709a,#fee140)',
]
function cover(i) { return coverColors[i % coverColors.length] }

async function load() {
  const r = await hotBlog(1)
  blogs.value = r.data || []
}

async function toggleLike(b) {
  try {
    await likeBlog(b.id)
    const liked = likedSet.value.has(b.id)
    if (liked) likedSet.value.delete(b.id)
    else likedSet.value.add(b.id)
    ElMessage.success(liked ? '已取消' : '点赞成功')
    await load()
  } catch (e) {}
}

async function showLikes(b) {
  const r = await blogLikes(b.id)
  likesMap.value = { ...likesMap.value, [b.id]: r.data || [] }
}

async function publish() {
  if (!dialog.title) { ElMessage.warning('请输入标题'); return }
  await saveBlog({ title: dialog.title, content: dialog.content, shopId: dialog.shopId })
  ElMessage.success('发布成功,已推送到粉丝动态')
  dialog.show = false
  dialog.title = ''; dialog.content = ''
  load()
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-title">探店笔记</div>
    <div class="page-sub">看看大家都在薅什么羊毛,也分享你的省钱攻略</div>

    <div class="blog-grid">
      <div v-for="(b, i) in blogs" :key="b.id" class="blog-card" @click="router.push(`/blog/${b.id}`)">
        <div class="blog-cover" :style="{ background: cover(i) }">
          <span class="blog-shop-tag" v-if="b.shopId">商铺 #{{ b.shopId }}</span>
        </div>
        <div class="blog-body">
          <h3>{{ b.title }}</h3>
          <p class="blog-content">{{ b.content || '这篇笔记很精彩,快来看看吧' }}</p>
          <div class="blog-author" @click.stop="router.push(`/profile/${b.userId}`)">
            <el-avatar :size="22" style="background:var(--brand)">{{ (b.authorName || 'U')[0] }}</el-avatar>
            <span class="link">{{ b.authorName || '用户' + b.userId }}</span>
          </div>
          <div class="blog-foot">
            <div :class="['like-btn', { liked: likedSet.has(b.id) }]" @click.stop="toggleLike(b)">
              <span class="heart">{{ likedSet.has(b.id) ? '❤️' : '🤍' }}</span>
              <span>{{ b.liked }}</span>
            </div>
            <el-button text size="small" @click.stop="showLikes(b)">点赞榜</el-button>
          </div>
          <transition name="fade">
            <div v-if="likesMap[b.id]" class="like-board">
              <div class="lb-title">点赞榜 Top{{ likesMap[b.id].length }}</div>
              <div class="lb-list">
                <span v-for="(u, idx) in likesMap[b.id]" :key="idx" class="lb-item">
                  {{ ['🥇','🥈','🥉'][idx] || '#' + (idx+1) }} 用户{{ u.userId }}
                </span>
              </div>
            </div>
          </transition>
        </div>
      </div>
      <el-empty v-if="!blogs.length" description="还没有笔记,快来发布第一篇" />
    </div>

    <!-- 发布悬浮按钮 -->
    <div class="fab" @click="dialog.show = true">
      <span>+</span>
    </div>

    <el-dialog v-model="dialog.show" title="发布探店笔记" width="500px">
      <el-form label-width="80px">
        <el-form-item label="商铺ID"><el-input-number v-model="dialog.shopId" :min="1" /></el-form-item>
        <el-form-item label="标题"><el-input v-model="dialog.title" placeholder="给笔记起个标题" /></el-form-item>
        <el-form-item label="正文">
          <el-input v-model="dialog.content" type="textarea" :rows="4" placeholder="分享你的省钱攻略..." />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialog.show = false">取消</el-button>
        <el-button type="primary" @click="publish">发布</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.blog-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.blog-card { background: #fff; border-radius: 12px; overflow: hidden; box-shadow: var(--shadow-sm); border: 1px solid var(--line); transition: all .2s; }
.blog-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }
.blog-cover { height: 130px; position: relative; }
.blog-shop-tag { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.4); color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 4px; }
.blog-body { padding: 14px; }
.blog-body h3 { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
.blog-content { color: var(--ink-3); font-size: 13px; line-height: 1.5; margin-bottom: 10px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.blog-author { display: flex; align-items: center; gap: 6px; font-size: 12px; margin-bottom: 10px; cursor: pointer; }
.blog-author:hover .link { color: var(--brand); }
.link { cursor: pointer; }
.blog-foot { display: flex; align-items: center; justify-content: space-between; }
.like-btn { display: flex; align-items: center; gap: 4px; cursor: pointer; padding: 4px 10px; border-radius: 14px; transition: all .2s; font-size: 13px; color: var(--ink-2); }
.like-btn:hover { background: var(--brand-light); }
.like-btn.liked { color: var(--brand); }
.like-btn.liked .heart { animation: pop 0.3s; }
@keyframes pop { 0% { transform: scale(1); } 50% { transform: scale(1.4); } 100% { transform: scale(1); } }
.like-board { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); }
.lb-title { font-size: 12px; color: var(--ink-3); margin-bottom: 6px; }
.lb-list { display: flex; flex-wrap: wrap; gap: 6px; }
.lb-item { font-size: 11px; background: var(--bg); padding: 2px 8px; border-radius: 10px; color: var(--ink-2); }

.fab { position: fixed; right: 32px; bottom: 32px; width: 52px; height: 52px; border-radius: 50%; background: var(--brand); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 28px; box-shadow: 0 6px 20px rgba(255,90,54,0.4); cursor: pointer; transition: transform .2s; z-index: 100; }
.fab:hover { transform: scale(1.1) rotate(90deg); }

.fade-enter-active, .fade-leave-active { transition: opacity .2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

@media (max-width: 900px) { .blog-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .blog-grid { grid-template-columns: 1fr; } }
</style>
