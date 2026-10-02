<script setup>
import { ref, reactive, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { agentChatStream, ragAsk, ragReindex } from '../api'
import { getToken } from '../utils/request'

const mode = ref('agent') // agent | rag
const input = ref('')
const sending = ref(false)
const messages = ref([]) // {role:'user'|'ai', text, steps:[]}
const chatBox = ref(null)

const suggestions = [
  '附近有什么便宜的奶茶?',
  '帮我规划一个 100 元的省钱周末',
  '现在有什么红包雨可以抢?',
  '帮我秒杀电影票',
  '火锅店有什么优惠?'
]

async function send(text) {
  const q = text || input.value
  if (!q.trim() || sending.value) return
  input.value = ''
  sending.value = true

  const msg = reactive({ role: 'ai', text: '', steps: [], typing: true })
  messages.value.push({ role: 'user', text: q, steps: [] })
  messages.value.push(msg)
  await scrollBottom()

  const token = getToken()
  try {
    if (mode.value === 'agent') {
      await agentChatStream(q, token, {
        onStep: async (s) => { msg.steps.push(s); await scrollBottom() },
        onToken: async (t) => { msg.text += t; await scrollBottom() },
        onDone: () => { msg.typing = false; sending.value = false },
        onError: (e) => { msg.text += '\n[错误: ' + e + ']'; msg.typing = false; sending.value = false }
      })
      msg.typing = false
      sending.value = false
    } else {
      // RAG 同步
      const r = await ragAsk(q)
      msg.text = r.data
      msg.steps.push('📚 已检索商铺/笔记向量化匹配')
      msg.typing = false
      sending.value = false
      await scrollBottom()
    }
  } catch (e) {
    msg.text += '\n[出错: ' + (e.msg || e.message) + ']'
    msg.typing = false
    sending.value = false
  }
}

async function scrollBottom() {
  await nextTick()
  if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}

async function reindex() {
  try {
    const r = await ragReindex()
    ElMessage.success('索引重建完成,共 ' + r.data.indexed + ' 条')
  } catch (e) {}
}
</script>

<template>
  <div class="page ai-page">
    <div class="page-title">AI 省钱助手</div>
    <div class="page-sub">智能查商铺、规划省钱方案、帮你秒杀抢红包</div>

    <div class="card chat-card">
      <div class="chat-head">
        <el-radio-group v-model="mode" size="small">
          <el-radio-button value="agent">Agent 模式(可调工具)</el-radio-button>
          <el-radio-button value="rag">RAG 问答(知识检索)</el-radio-button>
        </el-radio-group>
        <el-button size="small" @click="reindex" v-if="mode==='rag'">重建索引</el-button>
      </div>

      <div class="chat-box" ref="chatBox">
        <div v-if="!messages.length" class="empty">
          <div class="empty-icon">🤖</div>
          <p>你好,我是兜省省 AI 助手</p>
          <p class="muted">试试问我:</p>
          <div class="suggest">
            <el-button v-for="s in suggestions" :key="s" size="small" round @click="send(s)">{{ s }}</el-button>
          </div>
        </div>

        <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
          <div class="msg-avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
          <div class="msg-body">
            <div v-if="m.steps.length" class="steps">
              <div v-for="(s, j) in m.steps" :key="j" class="step">{{ s }}</div>
            </div>
            <div class="msg-text" v-if="m.text">
              <span v-if="m.typing" class="cursor">|</span>{{ m.text }}<span v-if="m.typing" class="cursor">|</span>
            </div>
          </div>
        </div>
      </div>

      <div class="chat-input">
        <el-input v-model="input" placeholder="问点什么..." @keyup.enter="send()" :disabled="sending" />
        <el-button type="primary" :loading="sending" @click="send()">发送</el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ai-page { max-width: 900px; }
.chat-card { display: flex; flex-direction: column; height: calc(100vh - 180px); min-height: 500px; }
.chat-head { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid var(--line); }
.chat-box { flex: 1; overflow-y: auto; padding: 16px 0; }
.empty { text-align: center; padding: 40px 20px; }
.empty-icon { font-size: 48px; margin-bottom: 12px; }
.suggest { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }

.msg { display: flex; gap: 10px; margin-bottom: 16px; }
.msg.user { flex-direction: row-reverse; }
.msg-avatar { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 600; }
.msg.ai .msg-avatar { background: var(--brand); color: #fff; }
.msg.user .msg-avatar { background: #4facfe; color: #fff; }
.msg-body { max-width: 75%; }
.msg.user .msg-body { text-align: right; }
.steps { margin-bottom: 8px; }
.step { font-size: 12px; color: var(--ink-3); background: var(--bg); padding: 4px 10px; border-radius: 6px; margin-bottom: 4px; text-align: left; }
.msg-text { display: inline-block; padding: 10px 14px; border-radius: 12px; text-align: left; white-space: pre-wrap; line-height: 1.6; }
.msg.ai .msg-text { background: #fff; border: 1px solid var(--line); }
.msg.user .msg-text { background: var(--brand); color: #fff; }
.cursor { animation: blink 1s infinite; color: var(--brand); }
@keyframes blink { 50% { opacity: 0; } }

.chat-input { display: flex; gap: 10px; padding-top: 12px; border-top: 1px solid var(--line); }
</style>
