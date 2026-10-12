import dayjs from 'dayjs'
import type { ReviewAction, SubmissionStatus } from './types'

/** 状态 / 动作的展示映射，三个页面共用，避免各处复制粘贴漂移 */
export const STATUS_LABEL: Record<SubmissionStatus, string> = { pending: '待审核', approved: '已通过', rejected: '已驳回' }
export const STATUS_TAG: Record<SubmissionStatus, 'warning' | 'success' | 'danger'> = {
  pending: 'warning',
  approved: 'success',
  rejected: 'danger',
}
export const STATUS_COLOR: Record<SubmissionStatus, string> = {
  pending: '#e6a23c',
  approved: '#67c23a',
  rejected: '#f56c6c',
}
export const ACTION_LABEL: Record<ReviewAction, string> = { approve: '通过', reject: '驳回', reopen: '重新审核' }
export const ACTION_TAG: Record<ReviewAction, 'success' | 'danger' | 'warning'> = {
  approve: 'success',
  reject: 'danger',
  reopen: 'warning',
}

/** 驳回弹窗中「自定义原因」选项的文案（预设列表本身来自后端 /api/meta） */
export const CUSTOM_REASON_LABEL = '其他'

/** 统一时间格式化：null 安全，full 控制是否带年份 */
export function fmtTime(iso: string | null, full = false): string {
  return iso ? dayjs(iso).format(full ? 'YYYY-MM-DD HH:mm' : 'MM-DD HH:mm') : '—'
}
