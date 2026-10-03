export type Role = 'user' | 'agent';
export type Group = 'user' | 'agent' | 'session';

export interface QuestionDef {
  id: string;
  type: 'choice' | 'noul' | 'score';
  label: string;
  instructions: string;
  criteria?: any;
}

export interface Policies {
  version: string;
  maxQuestionsPerRequest: number;
  groups: Record<Group, { label: string; desc: string; questions: QuestionDef[] }>;
}

/** 归一化后的单个判定结果 */
export interface Answer {
  type: 'choice' | 'noul' | 'score';
  /** choice=选中的标签 / noul=概率 0~1 / score=分值 */
  value: string | number;
  /** score 专用：分值对应的档位名 */
  level?: string;
  levels?: string[];
  confidence: number | null;
  probs: Record<string, number>;
}

export type Answers = Record<string, Answer>;

export interface JudgeMeta {
  engine: 'jev' | 'mock';
  model: string;
  requests: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  ms: number;
}

export interface JudgeResult {
  group: Group;
  answers: Answers;
  meta: JudgeMeta;
}

export interface Queue {
  id: string;
  name: string;
  desc: string;
  sla: string;
}

export interface Routing {
  queues: Queue[];
  rules: RoutingRule[];
  intentQueue: Record<string, string>;
  intercept: {
    minProb: number;
    blockLevels: Record<string, 'warn' | 'block'>;
    rewrite: Record<string, string>;
  };
}

export type Condition =
  | { q: string; in: string[] }
  | { q: string; gte: number }
  | { any: Condition[] };

export interface RoutingRule {
  id: string;
  name: string;
  when: Condition[];
  then: {
    queue: string;
    human: boolean;
    priority: 'P0' | 'P1' | 'P2' | 'P3';
    reason: string;
  };
}

export interface Decision {
  ruleId: string;
  ruleName: string;
  queueId: string;
  queueName: string;
  sla: string;
  human: boolean;
  priority: string;
  reason: string;
}

export interface Intercept {
  kind: string;
  level: 'warn' | 'block';
  prob: number;
  rewrite: string;
}

export interface Msg {
  id: string;
  role: Role;
  text: string;
  at: number;
  answers?: Answers;
  meta?: JudgeMeta;
  decision?: Decision;
  intercept?: Intercept;
  /** 被 block 级拦截后仍要看原文时用 */
  blocked?: boolean;
  pending?: boolean;
  error?: string;
}

export interface ScriptDef {
  id: string;
  name: string;
  tag: string;
  desc: string;
  messages: { role: Role; text: string }[];
}
