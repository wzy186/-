/** 行业 -> 分类色槽固定映射：颜色跟随实体而非排名；
    未知行业归入「未知行业」并使用中性灰（任务书允许只说明设计不实现）。 */

const INDUSTRY_COLORS: Record<string, string> = {
  科技: "var(--cat-1)",
  医疗: "var(--cat-2)",
  消费: "var(--cat-3)",
  能源: "var(--cat-4)",
  工业: "var(--cat-5)",
};

export function industryColor(name: string): string {
  return INDUSTRY_COLORS[name] ?? "var(--cat-other)";
}
