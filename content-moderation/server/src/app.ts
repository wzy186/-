import express from 'express';
import type { SQLInputValue } from 'node:sqlite';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getDb } from './db.js';
import { REJECT_REASONS } from './constants.js';

/** 构建 Express 应用（与启动逻辑分离，便于集成测试直接挂到随机端口） */
export function createApp(): express.Express {
  const app = express();
  app.use(express.json());

  // ---------- 时间工具（统计按本地自然日） ----------
  function dayKey(d: Date): string {
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${d.getFullYear()}-${m}-${dd}`;
  }
  function startOfDay(d: Date): Date {
    const x = new Date(d);
    x.setHours(0, 0, 0, 0);
    return x;
  }

  // ---------- 序列化 ----------
  // 查询时统一做列名别名，保证 toItem 拿到的都是 camelCase
  const SUB_COLUMNS = `
    id, title, content, author, status,
    image_urls AS imageUrls,
    reject_reason AS rejectReason,
    reviewed_by AS reviewedBy,
    reviewed_at AS reviewedAt,
    created_at AS createdAt
  `;
  // 日志列表查询的公共片段（/api/logs 与 /api/stats.recentLogs 共用）
  const LOG_SELECT = `
    SELECT l.id, l.submission_id AS submissionId, l.action, l.reason, l.operator,
           l.created_at AS createdAt, s.title AS submissionTitle
    FROM review_logs l
    JOIN submissions s ON s.id = l.submission_id
  `;
  interface SubRow {
    id: number; title: string; content: string; imageUrls: string; author: string;
    status: string; rejectReason: string | null; reviewedBy: string | null;
    reviewedAt: string | null; createdAt: string;
  }
  function toItem(row: SubRow) {
    let imageUrls: string[] = [];
    try {
      const parsed: unknown = JSON.parse(row.imageUrls);
      imageUrls = Array.isArray(parsed) ? parsed : [];
    } catch {
      imageUrls = [];
    }
    return {
      id: row.id,
      title: row.title,
      content: row.content,
      excerpt: row.content.replace(/\s+/g, ' ').slice(0, 80),
      imageUrls,
      author: row.author,
      status: row.status,
      rejectReason: row.rejectReason,
      reviewedBy: row.reviewedBy,
      reviewedAt: row.reviewedAt,
      createdAt: row.createdAt,
    };
  }

  function bad(res: express.Response, code: number, error: string) {
    res.status(code).json({ error });
  }

  /** 分页参数：非正整数（含小数、NaN）一律回退默认值 */
  function parsePositiveInt(raw: unknown, fallback: number, max?: number): number {
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 1) return fallback;
    return max ? Math.min(n, max) : n;
  }

  function operatorExists(name: string): boolean {
    return !!getDb().prepare('SELECT 1 AS x FROM operators WHERE name = ?').get(name);
  }

  /**
   * 应用一次通过/驳回（需在调用方开启的事务内执行）：
   * 只处理仍处于待审核的条目，返回是否实际更新。
   * 单条与批量共用，避免两处状态迁移逻辑漂移。
   */
  function applyReview(id: number, action: 'approve' | 'reject', reason: string | null, operator: string, now: string): boolean {
    const db = getDb();
    const row = db.prepare('SELECT status FROM submissions WHERE id = ?').get(id) as unknown as { status: string } | undefined;
    if (!row || row.status !== 'pending') return false;
    db.prepare(`UPDATE submissions SET status = ?, reject_reason = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?`)
      .run(action === 'approve' ? 'approved' : 'rejected', reason, operator, now, id);
    db.prepare(`INSERT INTO review_logs (submission_id, action, reason, operator, created_at) VALUES (?, ?, ?, ?, ?)`)
      .run(id, action, reason, operator, now);
    return true;
  }

  // ---------- 基础 ----------
  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  app.get('/api/operators', (_req, res) => {
    const db = getDb();
    res.json(db.prepare('SELECT id, name FROM operators ORDER BY id').all());
  });

  // 元数据（驳回原因预设等），前端从这取，避免前后端两份常量漂移
  app.get('/api/meta', (_req, res) => {
    res.json({ rejectReasons: REJECT_REASONS });
  });

  // ---------- 投稿 ----------
  app.get('/api/submissions', (req, res) => {
    const db = getDb();
    const status = String(req.query.status ?? 'pending');
    const q = String(req.query.q ?? '').trim();
    const sort = req.query.sort === 'oldest' ? 'ASC' : 'DESC';
    // page 也设上限：否则 (page-1)*pageSize 可能超出 SQLite OFFSET 的 64 位范围抛错
    const page = parsePositiveInt(req.query.page, 1, 1_000_000);
    const pageSize = parsePositiveInt(req.query.pageSize, 20, 100);

    const where: string[] = [];
    const params: SQLInputValue[] = [];
    if (status !== 'all') {
      where.push('status = ?');
      params.push(status);
    }
    if (q) {
      // 转义 LIKE 通配符，让用户搜索的 %/_ 按字面匹配
      const like = `%${q.replace(/[\\%_]/g, (m) => `\\${m}`)}%`;
      where.push(`(title LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\' OR author LIKE ? ESCAPE '\\')`);
      params.push(like, like, like);
    }
    const whereSql = where.length ? `WHERE ${where.join(' AND ')}` : '';

    const totalRow = db.prepare(`SELECT COUNT(*) AS n FROM submissions ${whereSql}`).get(...params) as unknown as { n: number };
    const rows = db
      .prepare(`SELECT ${SUB_COLUMNS} FROM submissions ${whereSql} ORDER BY created_at ${sort}, id ${sort} LIMIT ? OFFSET ?`)
      .all(...params, pageSize, (page - 1) * pageSize) as unknown as SubRow[];

    res.json({ items: rows.map(toItem), total: totalRow.n, page, pageSize });
  });

  app.get('/api/submissions/:id', (req, res) => {
    const db = getDb();
    const row = db.prepare(`SELECT ${SUB_COLUMNS} FROM submissions WHERE id = ?`).get(Number(req.params.id)) as unknown as SubRow | undefined;
    if (!row) return bad(res, 404, '投稿不存在');
    res.json(toItem(row));
  });

  /** 单个审核：action = approve | reject | reopen */
  app.post('/api/submissions/:id/review', (req, res) => {
    const id = Number(req.params.id);
    const { action, reason, operator } = (req.body ?? {}) as { action?: string; reason?: string; operator?: string };
    if (action !== 'approve' && action !== 'reject' && action !== 'reopen') return bad(res, 400, 'action 必须是 approve / reject / reopen');
    const op = (operator ?? '').trim();
    if (!op) return bad(res, 400, '缺少操作人 operator');

    const db = getDb();
    const row = db.prepare('SELECT status FROM submissions WHERE id = ?').get(id) as unknown as { status: string } | undefined;
    if (!row) return bad(res, 404, '投稿不存在');
    if ((action === 'approve' || action === 'reject') && row.status !== 'pending') {
      return bad(res, 409, '该投稿已不在待审核状态，请刷新列表');
    }
    if (action === 'reopen' && row.status === 'pending') return bad(res, 409, '该投稿本就处于待审核状态');
    if (action === 'reject' && !(reason ?? '').trim()) return bad(res, 400, '驳回必须填写原因');
    if (!operatorExists(op)) return bad(res, 400, '操作人不存在');

    const now = new Date().toISOString();
    db.exec('BEGIN');
    try {
      if (action === 'reopen') {
        db.prepare(`UPDATE submissions SET status = 'pending', reject_reason = NULL, reviewed_by = NULL, reviewed_at = NULL WHERE id = ?`).run(id);
        db.prepare(`INSERT INTO review_logs (submission_id, action, reason, operator, created_at) VALUES (?, ?, ?, ?, ?)`)
          .run(id, action, null, op, now);
      } else {
        applyReview(id, action, action === 'reject' ? (reason ?? null) : null, op, now);
      }
      db.exec('COMMIT');
    } catch (e) {
      db.exec('ROLLBACK');
      throw e;
    }
    const updated = db.prepare(`SELECT ${SUB_COLUMNS} FROM submissions WHERE id = ?`).get(id) as unknown as SubRow;
    res.json({ item: toItem(updated) });
  });

  /** 批量审核：action = approve | reject，只处理仍处于待审核的条目 */
  app.post('/api/submissions/batch-review', (req, res) => {
    const { ids, action, reason, operator } = (req.body ?? {}) as {
      ids?: unknown; action?: string; reason?: string; operator?: string;
    };
    if (!Array.isArray(ids) || ids.length === 0) return bad(res, 400, 'ids 必须是非空数组');
    if (ids.length > 500) return bad(res, 400, '单次批量操作最多 500 条');
    if (action !== 'approve' && action !== 'reject') return bad(res, 400, 'action 必须是 approve / reject');
    const op = (operator ?? '').trim();
    if (!op) return bad(res, 400, '缺少操作人 operator');
    if (action === 'reject' && !(reason ?? '').trim()) return bad(res, 400, '驳回必须填写原因');
    if (!operatorExists(op)) return bad(res, 400, '操作人不存在');

    const db = getDb();
    const now = new Date().toISOString();
    let updated = 0;
    let skipped = 0;
    db.exec('BEGIN');
    try {
      for (const raw of ids) {
        const id = Number(raw);
        if (!Number.isInteger(id)) {
          skipped++;
          continue;
        }
        if (applyReview(id, action, action === 'reject' ? (reason ?? null) : null, op, now)) updated++;
        else skipped++;
      }
      db.exec('COMMIT');
    } catch (e) {
      db.exec('ROLLBACK');
      throw e;
    }
    res.json({ updated, skipped });
  });

  // ---------- 操作日志 ----------
  app.get('/api/logs', (req, res) => {
    const db = getDb();
    const operator = String(req.query.operator ?? '').trim();
    const action = String(req.query.action ?? '').trim();
    const page = parsePositiveInt(req.query.page, 1, 1_000_000);
    const pageSize = parsePositiveInt(req.query.pageSize, 20, 100);

    const where: string[] = [];
    const params: SQLInputValue[] = [];
    if (operator) {
      where.push('l.operator = ?');
      params.push(operator);
    }
    if (['approve', 'reject', 'reopen'].includes(action)) {
      where.push('l.action = ?');
      params.push(action);
    }
    const whereSql = where.length ? `WHERE ${where.join(' AND ')}` : '';

    const totalRow = db
      .prepare(`SELECT COUNT(*) AS n FROM review_logs l ${whereSql}`)
      .get(...params) as unknown as { n: number };
    const rows = db
      .prepare(`
        ${LOG_SELECT}
        ${whereSql}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT ? OFFSET ?
      `)
      .all(...params, pageSize, (page - 1) * pageSize);

    res.json({ items: rows, total: totalRow.n, page, pageSize });
  });

  // ---------- 统计 ----------
  app.get('/api/stats', (_req, res) => {
    const db = getDb();

    const dist = { pending: 0, approved: 0, rejected: 0 };
    const statusRows = db.prepare('SELECT status, COUNT(*) AS n FROM submissions GROUP BY status').all() as unknown as Array<{ status: string; n: number }>;
    for (const r of statusRows) {
      if (r.status in dist) dist[r.status as keyof typeof dist] = r.n;
    }

    const dayStart = startOfDay(new Date());
    // 用 setDate 而非 +86400000ms 推日界，避免夏令时切换日窗口漂移
    const addDays = (d: Date, n: number) => {
      const x = new Date(d);
      x.setDate(x.getDate() + n);
      return x;
    };
    function actionCounts(sinceIso: string, untilIso: string) {
      const rows = db
        .prepare(`SELECT action, COUNT(*) AS n FROM review_logs WHERE action IN ('approve', 'reject') AND created_at >= ? AND created_at < ? GROUP BY action`)
        .all(sinceIso, untilIso) as unknown as Array<{ action: string; n: number }>;
      const approved = rows.find((r) => r.action === 'approve')?.n ?? 0;
      const rejected = rows.find((r) => r.action === 'reject')?.n ?? 0;
      return { approved, rejected, processed: approved + rejected };
    }
    const rate = (approved: number, processed: number) =>
      processed === 0 ? null : Math.round((approved / processed) * 1000) / 10;

    // 注意：Date 的极端值转出的 ISO 串无法直接按字典序比较，用固定边界字符串
    const overall = actionCounts('1970-01-01T00:00:00.000Z', '9999-12-31T23:59:59.999Z');

    const last7Days = [];
    for (let i = 6; i >= 0; i--) {
      const start = addDays(dayStart, -i);
      const c = actionCounts(start.toISOString(), addDays(start, 1).toISOString());
      last7Days.push({ date: dayKey(start), approved: c.approved, rejected: c.rejected });
    }
    // 今日即近 7 日窗口的最后一天，避免单独重复聚合
    const todayLast = last7Days[last7Days.length - 1];
    const today = { ...todayLast, processed: todayLast.approved + todayLast.rejected };

    const recentLogs = db
      .prepare(`
        ${LOG_SELECT}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT 5
      `)
      .all();

    res.json({
      pendingCount: dist.pending,
      today: { ...today, approveRate: rate(today.approved, today.processed) },
      overall: { ...overall, approveRate: rate(overall.approved, overall.processed) },
      statusDistribution: dist,
      last7Days,
      recentLogs,
    });
  });

  // 未匹配的 /api 路径统一返回 JSON 404
  app.use('/api', (_req, res) => bad(res, 404, '接口不存在'));

  // ---------- 生产模式：托管前端构建产物 ----------
  const webDist = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../web/dist');
  if (existsSync(webDist)) {
    app.use(express.static(webDist));
    // SPA 回退只兜路由，不吞静态资源 404（否则旧 hash 的 /assets/*.js 会拿到 index.html）
    app.get(/^\/(?!api(?:\/|$)|assets(?:\/|$)).*/, (_req, res) => {
      res.sendFile(path.join(webDist, 'index.html'));
    });
  }

  // ---------- 兜底错误处理 ----------
  app.use((err: Error & { status?: number; statusCode?: number }, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    // 客户端请求错误（express.json 的 400/413 等）如实返回，不伪装成 500
    const status = typeof err.status === 'number' ? err.status : typeof err.statusCode === 'number' ? err.statusCode : 500;
    if (status < 500) {
      res.status(status).json({ error: status === 413 ? '请求体过大' : '请求格式错误' });
      return;
    }
    console.error(err);
    res.status(500).json({ error: '服务器内部错误' });
  });

  return app;
}
