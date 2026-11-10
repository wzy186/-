/** 展示层格式化：仅负责千分位等纯呈现转换，不参与任何金额/占比计算。 */

/** "100000.00" -> "100,000.00"（千分位；后端保证两位小数） */
export function formatMoney(raw: string | null | undefined): string {
  if (raw === null || raw === undefined || raw === "") return "—";
  const negative = raw.startsWith("-");
  const body = negative ? raw.slice(1) : raw;
  const [intPart, decPart] = body.split(".");
  const grouped = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  const suffix = decPart !== undefined ? `.${decPart}` : "";
  return `${negative ? "-" : ""}${grouped}${suffix}`;
}

/** 数量千分位："1234567" -> "1,234,567" */
export function formatQty(raw: number): string {
  return String(raw).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
