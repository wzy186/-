import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  Answers,
  Group,
  JudgeMeta,
  Msg,
  Policies,
  Role,
  Routing,
  ScriptDef,
} from './types';
import { getConfig, getPolicies, getRouting, getScripts, judge } from './lib/api';
import { checkIntercept, decide } from './lib/decide';
import { MessageBubble } from './components/MessageBubble';
import { Dashboard } from './components/Dashboard';
import { Pill } from './components/ui';

const uid = () => Math.random().toString(36).slice(2, 10);

export default function App() {
  const [policies, setPolicies] = useState<Policies | null>(null);
  const [routing, setRouting] = useState<Routing | null>(null);
  const [scripts, setScripts] = useState<ScriptDef[]>([]);
  const [engine, setEngine] = useState<{ engine: 'jev' | 'mock'; model: string } | null>(null);
  const [bootError, setBootError] = useState('');

  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [role, setRole] = useState<Role>('user');
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState<string | null>(null);
  const [session, setSession] = useState<{ answers: Answers; meta: JudgeMeta } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem('cs-theme') || ''; } catch { return ''; }
  });

  const streamRef = useRef<HTMLDivElement>(null);
  const msgsRef = useRef<Msg[]>([]);
  msgsRef.current = msgs;
  const cancelRef = useRef(false);

  useEffect(() => {
    Promise.all([getPolicies(), getRouting(), getScripts(), getConfig()])
      .then(([p, r, s, c]) => {
        setPolicies(p);
        setRouting(r);
        setScripts(s.scripts);
        setEngine({ engine: c.engine, model: c.model });
      })
      .catch((e) => setBootError(String(e.message || e)));
  }, []);

  useEffect(() => {
    if (theme) document.documentElement.setAttribute('data-theme', theme);
    else document.documentElement.removeAttribute('data-theme');
  }, [theme]);

  useEffect(() => {
    const el = streamRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs]);

  const usage = useMemo(() => {
    const all = msgs.map((m) => m.meta).filter(Boolean) as JudgeMeta[];
    if (session?.meta) all.push(session.meta);
    return {
      requests: all.reduce((a, m) => a + m.requests, 0),
      inputTokens: all.reduce((a, m) => a + m.inputTokens, 0),
      costUsd: all.reduce((a, m) => a + m.costUsd, 0),
      ms: all.reduce((a, m) => a + m.ms, 0),
    };
  }, [msgs, session]);

  /** 发一条消息并跑判定。返回加进去的那条（已带判定结果）。 */
  const send = useCallback(
    async (text: string, who: Role) => {
      if (!text.trim() || !routing) return;
      const id = uid();
      const pending: Msg = { id, role: who, text: text.trim(), at: Date.now(), pending: true };
      setMsgs((prev) => [...prev, pending]);
      setBusy(true);
      const history = [...msgsRef.current, pending].map((m) => ({ role: m.role, text: m.text }));
      try {
        const res = await judge(who as Group, history);
        setMsgs((prev) =>
          prev.map((m) => {
            if (m.id !== id) return m;
            const answers = res.answers;
            const intercept = who === 'agent' ? checkIntercept(answers, routing) : null;
            const decision = who === 'user' ? decide(answers, routing) : undefined;
            return {
              ...m,
              pending: false,
              answers,
              meta: res.meta,
              decision: decision || undefined,
              intercept: intercept || undefined,
              blocked: intercept?.level === 'block',
            };
          })
        );
      } catch (e: any) {
        setMsgs((prev) =>
          prev.map((m) => (m.id === id ? { ...m, pending: false, error: String(e.message || e) } : m))
        );
      } finally {
        setBusy(false);
      }
    },
    [routing]
  );

  const onSend = async () => {
    const t = draft;
    setDraft('');
    await send(t, role);
  };

  const playScript = async (s: ScriptDef) => {
    cancelRef.current = false;
    setMsgs([]);
    setSession(null);
    setPlaying(s.id);
    for (const m of s.messages) {
      if (cancelRef.current) break;
      await send(m.text, m.role);
      await new Promise((r) => setTimeout(r, 260));
    }
    setPlaying(null);
  };

  const analyze = async () => {
    setAnalyzing(true);
    try {
      const res = await judge('session', msgs.map((m) => ({ role: m.role, text: m.text })));
      setSession({ answers: res.answers, meta: res.meta });
    } catch (e: any) {
      alert('会话分析失败：' + (e.message || e));
    } finally {
      setAnalyzing(false);
    }
  };

  if (bootError) {
    return (
      <div style={{ padding: 24 }}>
        <h2>启动失败</h2>
        <p className="muted">{bootError}</p>
        <p className="muted">后端没起来？dev 模式直接 <code>npm run dev</code>；生产模式先 build 再 <code>npm start</code>。</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="top">
        <h1>
          客服意图路由沙盘
          <small>意图路由 · 情绪监测 · 转人工判断 · 服务质检 · 危险话术拦截</small>
        </h1>
        <span className="spacer" />
        {engine && (
          <Pill tone={engine.engine === 'jev' ? 'ok' : 'warn'}>
            {engine.engine === 'jev' ? `Jev ${engine.model}` : 'Mock 规则引擎（未配置 API Key）'}
          </Pill>
        )}
        <button className="btn ghost tiny" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
          切换主题
        </button>
      </header>

      <div className="cols">
        {/* 左：剧本库 */}
        <div className="col">
          <div className="card sticky">
            <h2>剧本库</h2>
            <p className="hint">点一个自动逐条回放，看路由和拦截怎么跟着变</p>
            {scripts.map((s) => (
              <button
                key={s.id}
                className="script"
                onClick={() => playScript(s)}
                disabled={busy || playing !== null}
              >
                <span className="tag">{s.tag}</span>
                <b>{s.name}</b>
                <span>{s.desc}</span>
              </button>
            ))}
            <div className="row" style={{ marginTop: 8 }}>
              <button
                className="btn ghost tiny"
                onClick={() => {
                  cancelRef.current = true;
                  setMsgs([]);
                  setSession(null);
                  setPlaying(null);
                }}
              >
                清空会话
              </button>
            </div>
          </div>
        </div>

        {/* 中：对话 */}
        <div className="col">
          <div className="card chat">
            <h2>
              对话模拟 {playing && <Pill tone="accent">剧本回放中</Pill>}
            </h2>
            <p className="hint">
              切换身份发消息。用户消息触发意图/情绪/风险判定与路由；客服消息触发质检与危险话术拦截。
            </p>
            <div className="stream" ref={streamRef}>
              {msgs.length === 0 && (
                <div className="empty">
                  还没有消息。左边点一个剧本，或者直接在下面输入。
                </div>
              )}
              {msgs.map((m) => (
                <MessageBubble key={m.id} m={m} policies={policies} />
              ))}
            </div>

            <div className="composer">
              <div className="row" style={{ marginBottom: 8 }}>
                <div className="roleTabs">
                  <button aria-pressed={role === 'user'} onClick={() => setRole('user')}>
                    以用户身份
                  </button>
                  <button aria-pressed={role === 'agent'} onClick={() => setRole('agent')}>
                    以客服身份
                  </button>
                </div>
                <span className="tiny muted">
                  {role === 'user'
                    ? '会跑：一级意图 / 情绪 / 紧急度 / 转人工 / 用户风险'
                    : '会跑：危险话术 / 服务质量 / 共情 / 可执行方案'}
                </span>
              </div>
              <textarea
                rows={2}
                value={draft}
                placeholder={
                  role === 'user'
                    ? '例：我的退款三天了还没到账，你们到底什么时候处理？'
                    : '例：您放心，我跟您保证三天之内一定送到。'
                }
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) onSend();
                }}
              />
              <div className="row" style={{ marginTop: 8 }}>
                <button className="btn" onClick={onSend} disabled={busy || !draft.trim()}>
                  {busy ? '判定中…' : '发送并判定'}
                </button>
                <span className="tiny muted">⌘/Ctrl + Enter 发送</span>
              </div>
            </div>
          </div>
        </div>

        {/* 右：实时看板 */}
        <div className="col col-right">
          <Dashboard
            msgs={msgs}
            routing={routing}
            policies={policies}
            session={session}
            onAnalyze={analyze}
            analyzing={analyzing}
            usage={usage}
          />
        </div>
      </div>
    </div>
  );
}
