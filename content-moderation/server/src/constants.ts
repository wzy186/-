export const REVIEW_ACTIONS = ['approve', 'reject', 'reopen'] as const;
export type ReviewAction = (typeof REVIEW_ACTIONS)[number];

/** 驳回原因预设（前端通过 GET /api/meta 获取，保持单一事实源；「其他」由前端自行补充为自定义入口） */
export const REJECT_REASONS = ['广告营销', '内容低俗', '人身攻击', '敏感违规', '灌水无意义'] as const;
