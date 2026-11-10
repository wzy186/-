#!/usr/bin/env bash
# 验收截图生成脚本（VERIFY.md 插图）：
# 前置条件：后端已启动 http://127.0.0.1:8000，且已保存 AC-03 方案（id=1）。
# 用法：bash scripts/screenshots.sh
set -euo pipefail

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE="http://127.0.0.1:8000"
OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/screenshots"
mkdir -p "$OUT"

shot() { # $1=文件名 $2=窗口尺寸 $3=URL
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --window-size="$2" --virtual-time-budget=8000 \
    --screenshot="$OUT/$1" "$3" >/dev/null 2>&1
  echo "已生成 $OUT/$1"
}

shot 01-overview.png            1440,2100 "$BASE/?tab=overview"
shot 02-simulate-ac03.png       1440,1650 "$BASE/?tab=simulate&ops=SELL:A01:100,BUY:A02:100"
shot 03-error-ac04.png          1440,1650 "$BASE/?tab=simulate&ops=BUY:A01:201"
shot 04-saved-plan-ac05.png     1440,1550 "$BASE/?tab=plans&plan=1"
shot 05-rebalance-equal.png     1440,1700 "$BASE/?tab=simulate&rebalance=equal"
