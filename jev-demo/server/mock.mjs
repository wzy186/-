/**
 * Mock 判定引擎：没有 TYPESAFE_API_KEY 时用它，输出结构和 Jev 完全一致，
 * 所以前端不用知道自己在跟谁说话。
 *
 * 纯关键词加权 + 归一化，确定性输出（同样输入永远同样结果），方便做回归。
 * 它不是要取代模型，只是让这个工程开箱能演示、能写测试。
 */

const RE = (s) => new RegExp(s, 'i');

/** 命中一组词就加分：[正则, 权重] */
const INTENT = {
  物流查询: [[/物流|快递|发货|揽收|运单|单号|派送|到哪|什么时候(到|发)/, 3], [/查(一下|查)?件/, 2]],
  物流异常: [[/丢件|没收到|少发|漏发|错发|发错|空包|代签|被.{0,3}签收|签收了.{0,6}没/, 5], [/破损|裂缝|碎了|压坏/, 4]],
  退款退货: [[/退款|退货|仅退款|退钱|退回|到账/, 5]],
  换货补发: [[/换货|换一(个|件|台)|补发|换(尺码|颜色|个码)/, 5]],
  商品咨询: [[/支持|参数|材质|尺码|适配|兼容|型号|怎么用|有(没有)?货|hz|寸/, 4]],
  价格优惠: [[/多少钱|价格|优惠券|满减|便宜|保价|差价|活动价|降价/, 4]],
  订单修改: [[/改(地址|成|收货|手机)|修改订单|取消订单|催(一下)?发货|换地址/, 5]],
  支付问题: [[/扣款|扣了两次|重复扣|付款|支付|付不了|分期|没扣掉/, 5]],
  发票开票: [[/发票|开票|抬头|报销/, 5]],
  售后维修: [[/保修|维修|质保|上门|返修|申请售后|售后被/, 5], [/坏(了|的)|不能用|用不了/, 2]],
  账号问题: [[/登不(上|了)|登录|账号|绑定|验证码/, 5]],
  投诉举报: [[/投诉|举报|差评|处罚|给.{0,2}说法|要说法|什么破|答不上来|你们的态度/, 4]],
  活动规则: [[/活动|预售|定金|抽奖|积分|会员权益|规则/, 4]],
  闲聊其他: [[/^(你好|在吗|hi|hello)/, 3], [/谢谢|好的|辛苦了|多谢/, 3]],
};

const USER_RISK = {
  无: [],
  监管投诉: [[/12315|消协|市场监管|黑猫|小二|工商|举报到/, 6]],
  媒体曝光: [[/微博|抖音|小红书|记者|曝光|上热搜|发(到|个)?(视频|帖)/, 6]],
  法律诉讼: [[/起诉|律师|法院|三倍赔偿|法律途径|诉讼/, 6]],
  人身攻击: [[/傻|蠢|滚|废物|垃圾玩意|骂|你妈|神经病|狗/, 6]],
  疑似薅羊毛: [[/仅退款/, 3], [/不(用|想)?(寄|退)回/, 3], [/第.{0,2}(单|次)|这个月.{0,4}单/, 3], [/老规矩|都是直接退/, 3]],
};

const AGENT_RISK = {
  无风险: [],
  绝对化承诺: [[/保证|一定(能|会|可以|送|到)|绝对|百分百|100%|肯定不会|必然/, 6]],
  越权赔付: [[/赔(您|你|付|偿)|双倍|假一赔十|全额退款再|额外补偿|给您.{0,4}(块|元)/, 6]],
  时效硬承诺: [[/(\d+|三|两|一|二)(天|小时|个工作日)(之)?内.{0,6}(送到|到达|处理好|解决)/, 4], [/明天(必|一定)到/, 4]],
  泄露隐私: [[/1[3-9]\d{9}/, 8], [/签收人是|身份证|户名是/, 5], [/地址是.{0,10}(小区|号楼|室|栋)/, 5]],
  情绪失控: [[/您自己|你自己|这也要怪|说得很清楚|不讲理|爱信不信|有意思吗|我也没办法啊/, 6]],
  推诿甩锅: [[/不归我管|不是我们的(事|责任)|你(去)?找(快递|厂家|别人)|系统就(这样|是这样)|这是快递的事/, 6]],
  私下交易: [[/加(我)?微信|私下|走线下|转账给|返现|站外/, 6]],
  违规宣称: [[/治疗|根治|替代药|绝对安全|国家级|最高级|第一品牌/, 6]],
  诋毁他人: [[/比.{1,6}(强|好多了)|我们(公司|系统)也就|同事.{0,4}不行|平台垃圾/, 5]],
};

const EMOTION = [
  [/./, 0],
  [/又|已经.{0,4}(天|次)|催|急|怎么还|能不能快|麻烦.{0,2}快|还没.{0,4}(处理|回复|解决|给)/, 1],
  [/到底|每次|反复|是不是在|什么破|答不上来|太差|无语|敷衍|别再让我等/, 2],
  [/[!！]{2,}|给.{0,2}说法|不解决|我就|凭什么|受够了|忍不了|赶紧给我/, 3],
  [/投诉|曝光|起诉|12315|律师|骂|垃圾玩意|滚|傻|废物|你妈/, 4],
];

const URGENCY = [
  [/./, 0],
  [/尽快|快点|赶紧|着急|催|早点/, 1],
  [/今天|明天|周[一二三四五六日]|之前必须|必须(拿到|到)|截止|最后一天/, 2],
  [/手术|住院|赶(飞机|火车|高铁)|几小时|马上就|来不及/, 3],
];

const ESCALATE = [
  [/转人工|要人工|找主管|找你们领导/, 0.95],
  [/投诉|曝光|起诉|12315|赔偿/, 0.85],
  [/到底|每次|反复|已经.{0,4}天|不解决/, 0.7],
  [/丢件|没收到|破损|错发|重复扣|扣了两次/, 0.62],
];

const QUALITY_PLUS = [
  [/抱歉|理解|确实|不好意思|辛苦您|影响到您/, 1.0],
  [/帮您|为您|我(这边|来)|直接给您|已经(帮您)?(登记|记录|提交)/, 0.9],
  [/工号|单号|今天内|小时内(同步|回复)|随时找我|不用您/, 1.2],
];
const QUALITY_MINUS = [
  [/耐心等待|请您等待|按规则|系统显示就是|没办法/, 1.3],
  [/您自己|这也要怪|说得很清楚|不讲理/, 2.6],
  [/不归我管|你找|这是快递的事/, 2.0],
];

const EMPATHY = [/抱歉|理解|确实|不好意思|影响到您|辛苦您|心情/, 0];
const ACTIONABLE = [/帮您|为您|我来|直接给您|登记|补发|单号|工号|今天内|下一步|提交|加急/, 0];

const CHURN = [/不(会)?再(买|来)|卸载|取消会员|换(别家|其他家)|以后不|再也不/, 0];
const RESOLVED = [/谢谢|好的|可以了|解决了|辛苦了|这样挺好/, 0];

// ---------------------------------------------------------------- 工具

function hits(text, table) {
  let s = 0;
  for (const [re, w] of table) if (re.test(text)) s += w;
  return s;
}

function softmax(scores, temp = 1.1) {
  const keys = Object.keys(scores);
  const max = Math.max(...keys.map((k) => scores[k]));
  const exps = keys.map((k) => Math.exp((scores[k] - max) / temp));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  const out = {};
  keys.forEach((k, i) => (out[k] = exps[i] / sum));
  return out;
}

function topOf(probs) {
  return Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
}

function round(o, n = 3) {
  const out = {};
  for (const [k, v] of Object.entries(o)) out[k] = Number(v.toFixed(n));
  return out;
}

function choiceAnswer(text, table, options, baseKey, fallbackText) {
  const scores = {};
  for (const opt of options) {
    const t = table[opt] || [];
    scores[opt] = hits(text, t);
  }
  // 当前这句证据不足（比如「今天必须拿到」没提是什么事），就回头看上文
  if (fallbackText && Math.max(...options.map((o) => scores[o])) < 3) {
    for (const opt of options) scores[opt] += hits(fallbackText, table[opt] || []) * 0.6;
  }
  if (baseKey && scores[baseKey] !== undefined) {
    const others = Math.max(...options.filter((o) => o !== baseKey).map((o) => scores[o]));
    scores[baseKey] = others > 0 ? 0.2 : 4.5; // 没命中任何风险时「无」压倒性
  }
  const probs = softmax(scores);
  const [choice, p] = topOf(probs);
  return { type: 'choice', choice, probabilities: round(probs), confidence: Number(p.toFixed(3)) };
}

function scoreAnswer(text, ladder, levels) {
  const raw = ladder.map(([re, w]) => (re.test(text) ? w : 0));
  const idx = Math.max(...raw);
  const scores = {};
  levels.forEach((lv, i) => {
    scores[lv] = -Math.abs(i - idx) * 2.8;
  });
  const probs = softmax(scores, 0.6);
  const score = levels.reduce((acc, lv, i) => acc + i * probs[lv], 0);
  const legend = {};
  levels.forEach((lv, i) => (legend[String(i)] = lv));
  return {
    type: 'score',
    score: Number(score.toFixed(2)),
    legend,
    probabilities: round(probs),
    confidence: Number(topOf(probs)[1].toFixed(3)),
  };
}

function noulAnswer(text, table) {
  let p = 0.12;
  for (const [re, v] of table) if (re.test(text)) p = Math.max(p, v);
  return { type: 'noul', noul: Number(p.toFixed(3)), confidence: Number((0.6 + Math.abs(p - 0.5) * 0.7).toFixed(3)) };
}

function simpleNoul(text, re, yes = 0.88, no = 0.12) {
  return {
    type: 'noul',
    noul: re.test(text) ? yes : no,
    confidence: 0.82,
  };
}

// ---------------------------------------------------------------- 主入口

/**
 * @param {string} state 完整上下文（含历史），最后一行是待判定消息
 * @param {Array}  defs  问题定义
 * @param {string} focus 待判定的那条消息正文（判定主要看它）
 */
export function mockJudge(state, defs, focus) {
  const t0 = Date.now();
  const text = focus || state;
  const answers = {};
  for (const q of defs) {
    switch (q.id) {
      case 'intent':
        answers.intent = choiceAnswer(text, INTENT, Object.keys(q.criteria), null, state);
        break;
      case 'user_risk':
        answers.user_risk = choiceAnswer(text, USER_RISK, Object.keys(q.criteria), '无');
        break;
      case 'agent_risk':
        answers.agent_risk = choiceAnswer(text, AGENT_RISK, Object.keys(q.criteria), '无风险');
        break;
      case 'emotion':
        answers.emotion = scoreAnswer(text, EMOTION, q.criteria.map(labelOf));
        break;
      case 'urgency':
        answers.urgency = scoreAnswer(text, URGENCY, q.criteria.map(labelOf));
        break;
      case 'escalate':
        answers.escalate = noulAnswer(text, ESCALATE);
        break;
      case 'agent_quality': {
        const plus = QUALITY_PLUS.reduce((a, [re, w]) => a + (re.test(text) ? w : 0), 0);
        const minus = QUALITY_MINUS.reduce((a, [re, w]) => a + (re.test(text) ? w : 0), 0);
        const idx = Math.max(0, Math.min(4, Math.round(2 + plus - minus)));
        const levels = q.criteria.map(labelOf);
        answers.agent_quality = scoreAnswer('', [[/^$/, idx]], levels);
        break;
      }
      case 'empathy':
        answers.empathy = simpleNoul(text, EMPATHY[0]);
        break;
      case 'actionable':
        answers.actionable = simpleNoul(text, ACTIONABLE[0]);
        break;
      case 'churn':
        answers.churn = simpleNoul(state, CHURN[0], 0.86, 0.1);
        break;
      case 'resolved':
        answers.resolved = simpleNoul(state, RESOLVED[0], 0.78, 0.22);
        break;
      case 'csat': {
        const neg = hits(state, [[/投诉|曝光|起诉|垃圾|敷衍|太差/, 2], [/到底|每次|还没/, 1]]);
        const pos = hits(state, [[/谢谢|挺好|辛苦/, 2], [/抱歉|帮您|补发|工号/, 1]]);
        const idx = Math.max(0, Math.min(4, Math.round(2 + pos - neg)));
        answers.csat = scoreAnswer('', [[/^$/, idx]], q.criteria.map(labelOf));
        break;
      }
      case 'session_tag': {
        const opts = Object.keys(q.criteria);
        const scores = {};
        for (const o of opts) scores[o] = 0;
        if (/12315|曝光|起诉|微博/.test(state)) scores['合规风险'] += 4;
        if (/仅退款|老规矩|第.{0,2}单/.test(state)) scores['羊毛嫌疑'] += 5;
        if (/投诉|给个说法|不解决/.test(state)) scores['客诉升级'] += 4;
        if (/丢件|重复扣|系统|物流信息不更新/.test(state)) scores['系统故障'] += 2;
        if (/谢谢|挺好|解决了/.test(state)) scores['正常已解决'] += 4;
        if (Object.values(scores).every((v) => v === 0)) scores['待跟进'] += 3;
        const probs = softmax(scores);
        const [choice, p] = topOf(probs);
        answers.session_tag = {
          type: 'choice', choice, probabilities: round(probs), confidence: Number(p.toFixed(3)),
        };
        break;
      }
      default:
        answers[q.id] = { type: q.type, note: 'mock 未实现该问题' };
    }
  }
  return {
    answers,
    meta: {
      engine: 'mock',
      model: 'mock-rules-1.0',
      requests: Math.ceil(defs.length / 5),
      inputTokens: Math.round(state.length / 1.6),
      outputTokens: 0,
      costUsd: 0,
      ms: Date.now() - t0,
    },
  };
}

/** score 的档位可能写成「平静：正常陈述」，取冒号前的短标签 */
export function labelOf(s) {
  return String(s).split(/[:：]/)[0].trim();
}
