import axios from 'axios'
import { ElMessage } from 'element-plus'

const TOKEN_KEY = 'dss_token'
const USER_KEY = 'dss_user'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}
export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}
export function setUser(user) {
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}
export function getUser() {
  const s = localStorage.getItem(USER_KEY)
  return s ? JSON.parse(s) : null
}
export function isAdmin() {
  const u = getUser()
  return u && u.role === 1
}

const service = axios.create({
  baseURL: '/api',
  timeout: 10000
})

// 请求拦截:自动带 authorization header
service.interceptors.request.use(config => {
  const token = getToken()
  if (token) {
    config.headers['authorization'] = token
  }
  return config
})

// 响应拦截:统一处理 Result
service.interceptors.response.use(
  resp => {
    const r = resp.data
    if (r.success) {
      return r
    }
    ElMessage.error(r.msg || '请求失败')
    return Promise.reject(r)
  },
  err => {
    if (err.response && err.response.status === 401) {
      clearToken()
      ElMessage.warning('请先登录')
      location.hash = '#/login'
    } else {
      ElMessage.error(err.message || '网络异常')
    }
    return Promise.reject(err)
  }
)

export default service
