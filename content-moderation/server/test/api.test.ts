import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import type { Server } from 'node:http';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

// 必须在导入任何业务模块前指向独立临时库，避免污染开发数据
const tmpDir = mkdtempSync(path.join(tmpdir(), 'moderation-test-'));
process.env.MODERATION_DB_PATH = path.join(tmpDir, 'test.db');

const { createApp } = await import('../src/app.js');
const { resetAndSeed } = await import('../src/seed.js');

let server: Server;
let base: string;

before(async () => {
  resetAndSeed();
  server = createApp().listen(0);
  await new Promise<void>((resolve) => server.once('listening', () => resolve()));
  const addr = server.address() as { port: number };
  base = `http://127.0.0.1:${addr.port}`;
});

after(() => {
  server.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

async function jget(url: string): Promise<{ status: number; body: any }> {
  const r = await fetch(base + url);
  return { status: r.status, body: await r.json() };
}
async function jpost(url: string, body: unknown): Promise<{ status: number; body: any }> {
  const r = await fetch(base + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: r.status, body: await r.json() };
}
async function firstPendingId(): Promise<number> {
  const { status, body } = await jget('/api/submissions?status=pending&pageSize=1&sort=oldest');
  assert.equal(status, 200);
  return body.items[0].id as number;
}

test('基础接口：health / operators / meta / 未知 API 404', async () => {
  assert.deepEqual((await jget('/api/health')).body, { ok: true });

  const ops = await jget('/api/operators');
  assert.equal(ops.status, 200);
  assert.equal(ops.body.length, 4);
  assert.ok(ops.body.some((o: { name: string }) => o.name === '林晚晴'));

  const meta = await jget('/api/meta');
  assert.deepEqual(meta.body, { rejectReasons: ['广告营销', '内容低俗', '人身攻击', '敏感违规', '灌水无意义'] });

  const missing = await jget('/api/nope');
  assert.equal(missing.status, 404);
  assert.equal(missing.body.error, '接口不存在');
});

test('种子数据与统计：48 条、今日有数据、近 7 日每天有数据、时间戳合法', async () => {
  const { body: stats } = await jget('/api/stats');
  assert.equal(stats.pendingCount, 30);
  assert.equal(stats.statusDistribution.pending + stats.statusDistribution.approved + stats.statusDistribution.rejected, 48);
  assert.equal(stats.overall.processed, 18);
  assert.ok(stats.today.processed >= 5, `今日处理量应至少有 5 条种子数据，实际 ${stats.today.processed}`);
  assert.equal(stats.last7Days.length, 7);
  for (const day of stats.last7Days) {
    assert.ok(day.approved + day.rejected > 0, `近 7 日 ${day.date} 不应为空`);
  }

  const { body: all } = await jget('/api/submissions?status=all&pageSize=100');
  assert.equal(all.total, 48);
  for (const item of all.items) {
    if (item.reviewedAt) {
      assert.ok(item.reviewedAt > item.createdAt, `#${item.id} 审核时间应晚于提交时间`);
      assert.ok(item.reviewedBy, `#${item.id} 已审条目应有操作人`);
    }
  }
});

test('列表：分页、非法分页参数回退、LIKE 通配符按字面匹配', async () => {
  const p1 = await jget('/api/submissions?status=pending&pageSize=10');
  assert.equal(p1.status, 200);
  assert.equal(p1.body.items.length, 10);
  assert.equal(p1.body.total, 30);

  const frac = await jget('/api/submissions?pageSize=2.5&page=abc');
  assert.equal(frac.status, 200);
  assert.equal(frac.body.pageSize, 20);
  assert.equal(frac.body.page, 1);

  const huge = await jget('/api/submissions?status=all&page=500000000000000000');
  assert.equal(huge.status, 200);
  assert.deepEqual(huge.body.items, []);

  // % 与 _ 应作为字面量匹配，而不是通配符
  const percent = await jget(`/api/submissions?status=all&q=${encodeURIComponent('%')}`);
  assert.ok(percent.body.total < 48, `q=% 不应匹配全表，实际 ${percent.body.total}`);
  for (const item of percent.body.items) {
    assert.ok(
      item.title.includes('%') || item.content.includes('%') || item.author.includes('%'),
      'q=% 命中的条目应包含字面 %',
    );
  }
  const underscore = await jget(`/api/submissions?status=all&q=${encodeURIComponent('_')}`);
  assert.equal(underscore.body.total, 0);
});

test('单个审核生命周期：通过 → 409 → 重新审核 → 驳回（含校验）', async () => {
  const id = await firstPendingId();

  // 未知操作人
  const unknownOp = await jpost(`/api/submissions/${id}/review`, { action: 'approve', operator: '路人甲' });
  assert.equal(unknownOp.status, 400);

  // 通过
  const ok = await jpost(`/api/submissions/${id}/review`, { action: 'approve', operator: '林晚晴' });
  assert.equal(ok.status, 200);
  assert.equal(ok.body.item.status, 'approved');
  assert.equal(ok.body.item.reviewedBy, '林晚晴');

  // 重复审核
  const dup = await jpost(`/api/submissions/${id}/review`, { action: 'approve', operator: '林晚晴' });
  assert.equal(dup.status, 409);

  // 重新审核
  const reopened = await jpost(`/api/submissions/${id}/review`, { action: 'reopen', operator: '陈默' });
  assert.equal(reopened.status, 200);
  assert.equal(reopened.body.item.status, 'pending');
  assert.equal(reopened.body.item.reviewedBy, null);

  // 驳回缺原因
  const noReason = await jpost(`/api/submissions/${id}/review`, { action: 'reject', operator: '陈默' });
  assert.equal(noReason.status, 400);

  // 驳回（带原因）
  const rejected = await jpost(`/api/submissions/${id}/review`, { action: 'reject', reason: '广告营销 · 测试', operator: '陈默' });
  assert.equal(rejected.status, 200);
  assert.equal(rejected.body.item.status, 'rejected');
  assert.equal(rejected.body.item.rejectReason, '广告营销 · 测试');

  // 已驳回再通过
  const again = await jpost(`/api/submissions/${id}/review`, { action: 'approve', operator: '陈默' });
  assert.equal(again.status, 409);
});

test('批量审核：只处理待审核条目，其余跳过并计数', async () => {
  const p = await jget('/api/submissions?status=pending&pageSize=3');
  const pendingIds: number[] = p.body.items.map((x: { id: number }) => x.id);
  assert.equal(pendingIds.length, 3);

  const done = await jget('/api/submissions?status=approved&pageSize=1');
  const approvedId: number = done.body.items[0].id;

  const r = await jpost('/api/submissions/batch-review', {
    ids: [...pendingIds, approvedId],
    action: 'approve',
    operator: '苏小满',
  });
  assert.equal(r.status, 200);
  assert.equal(r.body.updated, 3);
  assert.equal(r.body.skipped, 1);

  // 超上限
  const huge = await jpost('/api/submissions/batch-review', {
    ids: new Array(501).fill(1),
    action: 'approve',
    operator: '苏小满',
  });
  assert.equal(huge.status, 400);
});

test('操作日志：筛选与完整性', async () => {
  const all = await jget('/api/logs?pageSize=100');
  assert.ok(all.body.total >= 21, `种子 18 条 + 本组动作应产生至少 21 条日志，实际 ${all.body.total}`);

  const byOp = await jget(`/api/logs?operator=${encodeURIComponent('林晚晴')}&pageSize=100`);
  assert.ok(byOp.body.total >= 1);
  for (const log of byOp.body.items) assert.equal(log.operator, '林晚晴');

  const reopens = await jget('/api/logs?action=reopen&pageSize=100');
  assert.ok(reopens.body.total >= 1);
  for (const log of reopens.body.items) assert.equal(log.action, 'reopen');
});

test('统计随操作变化；坏 JSON 返回 400', async () => {
  const { body: stats } = await jget('/api/stats');
  // 本文件已完成：1 条通过 + 3 条批量通过（另有 1 条通过→重新审核→驳回）
  assert.equal(stats.pendingCount, 26);
  assert.ok(stats.overall.processed > 18, '操作后累计处理量应增加');

  const bad = await fetch(`${base}/api/submissions/1/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{bad json',
  });
  assert.equal(bad.status, 400);
});

test('新建投稿：校验、进入待审核队列、可被审核', async () => {
  // 必填校验
  const noTitle = await jpost('/api/submissions', { content: 'x', author: '张三' });
  assert.equal(noTitle.status, 400);
  const noAuthor = await jpost('/api/submissions', { title: '标题', content: '正文' });
  assert.equal(noAuthor.status, 400);

  // 图片校验：非 http(s) 链接、超过 4 张
  const badUrl = await jpost('/api/submissions', {
    title: 't', content: 'c', author: 'a', imageUrls: ['ftp://x'],
  });
  assert.equal(badUrl.status, 400);
  const tooMany = await jpost('/api/submissions', {
    title: 't', content: 'c', author: 'a',
    imageUrls: ['https://a', 'https://b', 'https://c', 'https://d', 'https://e'],
  });
  assert.equal(tooMany.status, 400);

  // 成功创建 → 待审核 → 出现在队列最前（最新提交排前）
  const created = await jpost('/api/submissions', {
    title: '测试投稿：新品发布会', content: '正文内容', author: '测试用户',
    imageUrls: ['https://picsum.photos/seed/test/600/400'],
  });
  assert.equal(created.status, 201);
  assert.equal(created.body.item.status, 'pending');
  assert.equal(created.body.item.author, '测试用户');
  assert.deepEqual(created.body.item.imageUrls, ['https://picsum.photos/seed/test/600/400']);

  const p = await jget('/api/submissions?status=pending&pageSize=1');
  assert.equal(p.body.items[0].id, created.body.item.id);

  // 闭环：可被正常审核
  const r = await jpost(`/api/submissions/${created.body.item.id}/review`, { action: 'approve', operator: '林晚晴' });
  assert.equal(r.status, 200);
  assert.equal(r.body.item.status, 'approved');
});
