import type { Operator, PageResult, ReviewAction, ReviewLog, Stats, SubmissionItem, SubmissionStatus } from './types'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  let data: unknown = null
  try {
    data = await res.json()
  } catch {
    // 非 JSON 响应（如被代理替换的 HTML 页面），同样按失败处理
  }
  if (!res.ok || data === null) {
    const msg =
      data !== null && typeof data === 'object' && 'error' in data
        ? String((data as { error: string }).error)
        : `请求失败（HTTP ${res.status}）`
    throw new Error(msg)
  }
  return data as T
}

function post<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export interface SubmissionQuery {
  status: SubmissionStatus | 'all'
  q?: string
  sort?: 'newest' | 'oldest'
  page?: number
  pageSize?: number
}

export const api = {
  operators: () => request<Operator[]>('/api/operators'),

  meta: () => request<{ rejectReasons: string[] }>('/api/meta'),

  submissions: (query: SubmissionQuery) => {
    const params = new URLSearchParams()
    params.set('status', query.status)
    if (query.q) params.set('q', query.q)
    params.set('sort', query.sort ?? 'newest')
    params.set('page', String(query.page ?? 1))
    params.set('pageSize', String(query.pageSize ?? 20))
    return request<PageResult<SubmissionItem>>(`/api/submissions?${params}`)
  },

  submission: (id: number) => request<SubmissionItem>(`/api/submissions/${id}`),

  createSubmission: (body: { title: string; content: string; author: string; imageUrls: string[] }) =>
    post<{ item: SubmissionItem }>('/api/submissions', body),

  review: (id: number, body: { action: ReviewAction; reason?: string; operator: string }) =>
    post<{ item: SubmissionItem }>(`/api/submissions/${id}/review`, body),

  batchReview: (body: { ids: number[]; action: 'approve' | 'reject'; reason?: string; operator: string }) =>
    post<{ updated: number; skipped: number }>('/api/submissions/batch-review', body),

  logs: (query: { operator?: string; action?: ReviewAction | ''; page?: number; pageSize?: number }) => {
    const params = new URLSearchParams()
    if (query.operator) params.set('operator', query.operator)
    if (query.action) params.set('action', query.action)
    params.set('page', String(query.page ?? 1))
    params.set('pageSize', String(query.pageSize ?? 20))
    return request<PageResult<ReviewLog>>(`/api/logs?${params}`)
  },

  stats: () => request<Stats>('/api/stats'),
}
