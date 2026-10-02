<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { follow, isFollow, commonFollows, feed } from '../api'

const router = useRouter()

const target = ref(2)
const followed = ref(false)
const common = ref([])
const feedList = ref([])
const feedState = reactive({ max: 0, offset: 0 })

async function checkFollow() {
  const r = await isFollow(target.value)
  followed.value = r.data
}

async function toggleFollow() {
  await follow(target.value, !followed.value)
  followed.value = !followed.value
  ElMessage.success(followed.value ? '关注成功' : '已取消关注')
  if (followed.value) loadCommon()
}

async function loadCommon() {
  const r = await commonFollows(target.value)
  common.value = r.data || []
}

async function loadFeed() {
  const r = await feed(feedState.max, feedState.offset)
  const d = r.data
  feedList.value = d.list || []
  feedState.max = d.minTime
  feedState.offset = d.offset
}

onMounted(() => { checkFollow(); loadFeed() })
</script>

<template>
  <div class="page">
    <div class="page-title">我的关注</div>
    <div class="page-sub">关注达人,在动态流看到他们的最新省钱分享</div>

    <div class="follow-layout">
      <!-- 左:用户关注卡 -->
      <div>
        <div class="user-card">
          <div class="uc-bg"></div>
          <div class="uc-body">
            <el-avatar :size="64" style="background:var(--brand);font-size:24px;">U{{ target }}</el-avatar>
            <h2>用户 {{ target }}</h2>
            <p class="muted">省钱达人 · 热爱分享</p>
            <el-button :type="followed ? 'default' : 'primary'" size="large" round @click="toggleFollow"
                       :style="followed ? {} : { background: 'var(--brand)' }">
              {{ followed ? '已关注' : '+ 关注' }}
            </el-button>
          </div>
        </div>

        <div class="card" v-if="common.length">
          <div class="card-title">
            <h3>共同关注</h3>
            <span class="muted">{{ common.length }} 人</span>
          </div>
          <div class="common-list">
            <div v-for="uid in common" :key="uid" class="common-item" @click="router.push(`/profile/${uid}`)">
              <el-avatar :size="36" style="background:var(--brand-light);color:var(--brand)">U{{ uid }}</el-avatar>
              <span class="link">用户 {{ uid }}</span>
            </div>
          </div>
        </div>
        <div class="card">
          <el-input-number v-model="target" :min="1" @change="checkFollow" />
          <span class="muted" style="margin-left:8px">输入用户 ID 切换</span>
          <el-button @click="loadCommon" style="margin-left:8px">查共同关注</el-button>
        </div>
      </div>

      <!-- 右:动态 Feed 流 -->
      <div class="card feed-card">
        <div class="card-title">
          <h3>动态收件箱</h3>
          <el-button size="small" @click="loadFeed">刷新</el-button>
        </div>
        <p class="muted" style="margin-bottom:16px">你关注的人发布的最新笔记</p>

        <el-empty v-if="!feedList.length" description="收件箱为空,关注别人后,对方发笔记你才会收到" :image-size="100" />

        <div v-else class="timeline">
          <div v-for="(id, i) in feedList" :key="i" class="tl-item">
            <div class="tl-dot"></div>
            <div class="tl-card">
              <div class="tl-head">
                <el-avatar :size="28" style="background:var(--brand)">U</el-avatar>
                <span class="tl-author">关注的人</span>
                <span class="muted">发布了笔记</span>
              </div>
              <div class="tl-content link" @click="router.push(`/blog/${id}`)">📝 笔记 #{{ id }} →</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.follow-layout { display: grid; grid-template-columns: 320px 1fr; gap: 16px; align-items: start; }

.user-card { background: #fff; border-radius: 14px; overflow: hidden; box-shadow: var(--shadow-sm); margin-bottom: 16px; }
.uc-bg { height: 80px; background: linear-gradient(135deg, #FF5A36, #FF8A5C); }
.uc-body { text-align: center; padding: 0 20px 24px; margin-top: -32px; position: relative; }
.uc-body h2 { font-size: 18px; margin: 10px 0 4px; }
.uc-body .muted { margin-bottom: 16px; }

.common-list { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.common-item { display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 12px; color: var(--ink-2); cursor: pointer; }
.common-item:hover { color: var(--brand); }

.feed-card { min-height: 400px; }
.timeline { position: relative; padding-left: 20px; }
.timeline::before { content: ''; position: absolute; left: 5px; top: 8px; bottom: 8px; width: 2px; background: var(--line); }
.tl-item { position: relative; margin-bottom: 14px; }
.tl-dot { position: absolute; left: -19px; top: 14px; width: 12px; height: 12px; border-radius: 50%; background: var(--brand); border: 2px solid #fff; box-shadow: 0 0 0 2px var(--brand); }
.tl-card { background: var(--bg); border-radius: 10px; padding: 12px 14px; }
.tl-head { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.tl-author { font-weight: 600; color: var(--ink); }
.tl-content { margin-top: 8px; padding: 10px; background: #fff; border-radius: 8px; font-size: 14px; }
.tl-content.link, .link { cursor: pointer; }
.tl-content.link:hover { color: var(--brand); }

@media (max-width: 900px) {
  .follow-layout { grid-template-columns: 1fr; }
}
</style>
