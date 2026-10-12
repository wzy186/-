export type SubmissionStatus = 'pending' | 'approved' | 'rejected'
export type ReviewAction = 'approve' | 'reject' | 'reopen'

export interface SubmissionItem {
  id: number
  title: string
  content: string
  excerpt: string
  imageUrls: string[]
  author: string
  status: SubmissionStatus
  rejectReason: string | null
  reviewedBy: string | null
  reviewedAt: string | null
  createdAt: string
}

export interface Operator {
  id: number
  name: string
}

export interface ReviewLog {
  id: number
  submissionId: number
  submissionTitle: string
  action: ReviewAction
  reason: string | null
  operator: string
  createdAt: string
}

export interface Stats {
  pendingCount: number
  today: { approved: number; rejected: number; processed: number; approveRate: number | null }
  overall: { approved: number; rejected: number; processed: number; approveRate: number | null }
  statusDistribution: { pending: number; approved: number; rejected: number }
  last7Days: Array<{ date: string; approved: number; rejected: number }>
  recentLogs: Array<{
    id: number
    action: ReviewAction
    operator: string
    createdAt: string
    submissionTitle: string
    reason: string | null
  }>
}

export interface PageResult<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
}
