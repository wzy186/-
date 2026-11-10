/** 触发状态徽标：图标 + 文字，颜色不单独表意。 */

export default function TriggerBadge({ triggered }: { triggered: boolean }) {
  return triggered ? (
    <span className="badge badge-alert">⚠ 超限</span>
  ) : (
    <span className="badge badge-ok">✓ 正常</span>
  );
}
