/** 全局错误横幅（组合加载失败等），带重试按钮。 */

export default function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-banner">
      <span>⚠</span>
      <span>{message}</span>
      {onRetry && (
        <>
          <span className="spacer" />
          <button className="btn btn-sm" onClick={onRetry}>
            重试
          </button>
        </>
      )}
    </div>
  );
}
