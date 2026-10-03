import type { Group, JudgeResult, Policies, Routing, ScriptDef } from '../types';

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export const getConfig = () =>
  get<{ engine: 'jev' | 'mock'; model: string; hasKey: boolean; maxQuestionsPerRequest: number }>(
    '/api/config'
  );
export const getPolicies = () => get<Policies>('/api/policies');
export const getRouting = () => get<Routing>('/api/routing');
export const getScripts = () => get<{ scripts: ScriptDef[] }>('/api/scripts');

export async function judge(
  group: Group,
  messages: { role: string; text: string }[],
  opts: { forceMock?: boolean } = {}
): Promise<JudgeResult> {
  const r = await fetch('/api/judge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group, messages, ...opts }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || `判定失败 ${r.status}`);
  return j;
}
