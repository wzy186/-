import { ref } from 'vue'
import { api } from './api'

// 值班场景不做账号体系：右上角选一下「我是谁」即可，
// 选择存在 localStorage，操作时随请求带给后端记录
const STORAGE_KEY = 'moderation-operator'

export const currentOperator = ref<string>(localStorage.getItem(STORAGE_KEY) ?? '')

export function setOperator(name: string): void {
  currentOperator.value = name
  localStorage.setItem(STORAGE_KEY, name)
}

/** 驳回原因预设：来自后端 /api/meta（单一事实源），页面内缓存 */
export const rejectReasons = ref<string[]>([])

export async function loadRejectReasons(): Promise<string[]> {
  if (!rejectReasons.value.length) {
    try {
      rejectReasons.value = (await api.meta()).rejectReasons
    } catch {
      // 加载失败保持空列表：驳回弹窗仍可用「其他」自定义原因，不产生未处理拒绝
    }
  }
  return rejectReasons.value
}
