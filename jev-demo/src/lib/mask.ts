/**
 * 脱敏。看板要展示「刚刚被拦下来的那句话」，如果原样显示，
 * 等于拦截了一次又在另一个地方泄露一次。所以凡是进看板的文本都过这里。
 */
const RULES: [RegExp, (m: string) => string][] = [
  [/1[3-9]\d{9}/g, (m) => `${m.slice(0, 3)}****${m.slice(-4)}`],
  [/\d{17}[\dXx]|\d{15}/g, (m) => `${m.slice(0, 4)}**********${m.slice(-2)}`], // 身份证
  [/\d{16,19}/g, (m) => `****${m.slice(-4)}`], // 银行卡
  [/[\w.+-]+@[\w-]+\.[\w.]+/g, (m) => m.replace(/^(.).*(@.*)$/, '$1***$2')],
  [/(省|市|区|县|镇|街道)[^，。；,;]{0,16}?\d+\s*(号楼|栋|单元|室|号)(\s*\d+)?/g, () => '［地址已隐藏］'],
];

export function mask(text: string): string {
  let out = text;
  for (const [re, fn] of RULES) out = out.replace(re, (m) => fn(m));
  return out;
}
