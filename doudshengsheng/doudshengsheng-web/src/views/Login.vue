<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { sendCode, login, me } from '../api'
import { setUser, setToken, getUser } from '../utils/request'

const router = useRouter()
const form = reactive({ phone: '', code: '' })
const countdown = ref(0)
const loading = ref(false)

if (getUser()) router.push('/')

async function sendSms() {
  if (!/^1\d{10}$/.test(form.phone)) {
    ElMessage.warning('请输入正确的手机号')
    return
  }
  try {
    await sendCode(form.phone)
    ElMessage.success('验证码已发送')
    countdown.value = 60
    const t = setInterval(() => {
      countdown.value--
      if (countdown.value <= 0) clearInterval(t)
    }, 1000)
  } catch (e) {}
}

async function doLogin() {
  if (!form.phone || !form.code) {
    ElMessage.warning('请输入手机号和验证码')
    return
  }
  loading.value = true
  try {
    const r = await login(form)
    setToken(r.data)
    const u = await me()
    if (u.data) setUser(u.data)
    ElMessage.success('登录成功')
    router.push('/')
  } catch (e) {} finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login">
    <!-- 左侧品牌区 -->
    <div class="brand">
      <div class="brand-inner">
        <div class="logo">兜省省</div>
        <h1>每一天,都省一点</h1>
        <p class="slogan">精选本地优惠 · 红包雨天天领 · 限时秒杀不停</p>
        <div class="features">
          <div class="feat"><span class="dot"></span> 附近优惠,按距离推荐</div>
          <div class="feat"><span class="dot"></span> 红包雨,抢到就是赚到</div>
          <div class="feat"><span class="dot"></span> 限时秒杀,手快有手慢无</div>
        </div>
      </div>
      <div class="brand-bg"></div>
    </div>

    <!-- 右侧登录区 -->
    <div class="form-side">
      <div class="form-box">
        <h2>欢迎回来</h2>
        <p class="form-sub">登录后开启你的省钱之旅</p>
        <el-form @submit.prevent style="margin-top: 24px">
          <el-form-item>
            <el-input v-model="form.phone" placeholder="手机号" size="large" prefix-icon="Iphone" />
          </el-form-item>
          <el-form-item>
            <div class="row" style="width:100%">
              <el-input v-model="form.code" placeholder="验证码" size="large" prefix-icon="Key" style="flex:1" />
              <el-button :disabled="countdown > 0" size="large" @click="sendSms">
                {{ countdown > 0 ? `${countdown}s` : '获取验证码' }}
              </el-button>
            </div>
          </el-form-item>
          <el-button type="primary" :loading="loading" size="large" style="width:100%; margin-top:8px" @click="doLogin">
            登录 / 注册
          </el-button>
        </el-form>
        <p class="terms">登录即代表同意《用户协议》和《隐私政策》</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.login { display: flex; height: 100%; }

/* 左侧品牌 */
.brand { flex: 1; position: relative; overflow: hidden; background: linear-gradient(135deg, #FF5A36 0%, #FF8A5C 100%); display: flex; align-items: center; }
.brand-inner { position: relative; z-index: 2; padding: 60px; color: #fff; max-width: 480px; }
.logo { font-size: 26px; font-weight: 700; letter-spacing: 2px; margin-bottom: 32px; opacity: .95; }
.brand h1 { font-size: 38px; font-weight: 700; line-height: 1.3; margin-bottom: 14px; }
.slogan { font-size: 15px; opacity: .9; line-height: 1.7; margin-bottom: 40px; }
.features { display: flex; flex-direction: column; gap: 14px; }
.feat { display: flex; align-items: center; gap: 10px; font-size: 14px; opacity: .95; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #fff; opacity: .9; }
/* 装饰圆 */
.brand-bg { position: absolute; inset: 0; z-index: 1; }
.brand-bg::before, .brand-bg::after { content: ''; position: absolute; border-radius: 50%; background: rgba(255,255,255,.12); }
.brand-bg::before { width: 400px; height: 400px; right: -120px; top: -120px; }
.brand-bg::after { width: 260px; height: 260px; left: -80px; bottom: -80px; background: rgba(255,255,255,.08); }

/* 右侧表单 */
.form-side { width: 460px; background: #fff; display: flex; align-items: center; justify-content: center; }
.form-box { width: 100%; max-width: 340px; padding: 40px; }
.form-box h2 { font-size: 24px; font-weight: 600; }
.form-sub { color: var(--ink-3); font-size: 14px; margin-top: 6px; }
.terms { color: var(--ink-3); font-size: 12px; margin-top: 20px; text-align: center; }

@media (max-width: 768px) {
  .brand { display: none; }
  .form-side { width: 100%; }
}
</style>
