#!/usr/bin/env bash
# 秒杀并发压测 + 超卖验证
# 用法:bash seckill_bench.sh [用户数] [初始库存]
# 流程:注册 N 个用户 → 重置库存 → 并发秒杀 → 统计 QPS + 验证无超卖
set -e

USERS=${1:-300}
STOCK=${2:-200}
BASE=http://localhost:8081
VOUCHER=10

echo "=== 秒杀压测: $USERS 用户并发抢券 $VOUCHER,初始库存 $STOCK ==="

# 1. 注册用户拿 token
echo "准备 $USERS 个用户 token..."
TOKENS=()
for i in $(seq 1 $USERS); do
  phone=$(printf "139%08d" $((10000 + i)))
  curl -s --max-time 5 -X POST "$BASE/user/code?phone=$phone" >/dev/null
  code=$(redis-cli GET "dss:login:code:$phone")
  tk=$(curl -s --max-time 5 -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$phone\",\"code\":\"$code\"}" | python3 -c "import sys,json;print(json.load(sys.stdin).get('data',''))")
  TOKENS+=("$tk")
  if [ $((i % 50)) -eq 0 ]; then echo "  已准备 $i / $USERS"; fi
done
echo "已准备 ${#TOKENS[@]} 个 token"

# 2. 重置库存 + 清已购记录 + 清订单 + 清限流器(压测用,避免被旧限流配置挡)
echo "重置库存到 $STOCK..."
redis-cli SET "dss:seckill:stock:$VOUCHER" $STOCK >/dev/null
redis-cli DEL "dss:seckill:order:$VOUCHER" >/dev/null
redis-cli --scan --pattern "dss:ratelimit*seckill*" | xargs -r redis-cli DEL >/dev/null 2>&1
mysql -uroot doudshengsheng -e "DELETE FROM tb_voucher_order WHERE voucher_id=$VOUCHER;" 2>/dev/null

# 3. 并发秒杀(每个 token 一个后台进程)
echo "开始并发秒杀..."
TMP=$(mktemp -d)
START=$(python3 -c 'import time;print(time.time())')
for tk in "${TOKENS[@]}"; do
  (
    resp=$(curl -s -X POST -H "authorization: $tk" "$BASE/voucher/order/seckill/$VOUCHER")
    ok=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(1 if d.get('success') else 0)" 2>/dev/null)
    msg=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('msg','')[:20])" 2>/dev/null)
    echo "$ok|$msg" >> "$TMP/results.txt"
  ) &
done
wait
END=$(python3 -c 'import time;print(time.time())')

# 4. 统计
OK=$(grep -c '^1|' "$TMP/results.txt")
FAIL=$(grep -c '^0|' "$TMP/results.txt")
COST=$(python3 -c "print(round($END-$START, 3))")
QPS=$(python3 -c "print(round($USERS/$COST))")

echo ""
echo "=== 结果 ==="
echo "总用户: $USERS"
echo "成功下单: $OK"
echo "失败: $FAIL"
echo "耗时: ${COST}s"
echo "QPS: $QPS"
echo "失败原因分布:"
cut -d'|' -f2 "$TMP/results.txt" | sort | uniq -c | sort -rn | head -5

# 5. 超卖验证
REMAIN=$(redis-cli GET "dss:seckill:stock:$VOUCHER")
echo ""
echo "=== 超卖验证 ==="
echo "初始库存: $STOCK"
echo "成功下单: $OK"
echo "Redis 剩余: $REMAIN"
EXPECTED=$((STOCK - OK))
if [ "$REMAIN" = "$EXPECTED" ]; then
  echo "✅ 无超卖: 初始($STOCK) - 成功($OK) == 剩余($REMAIN)"
else
  echo "❌ 可能超卖: 初始($STOCK) - 成功($OK) = $EXPECTED,但剩余 $REMAIN"
fi

# 6. DB 落库校验(等异步消费)
echo ""
echo "等 3 秒让异步落库..."
sleep 3
DB_CNT=$(mysql -uroot doudshengsheng -e "SELECT COUNT(*) FROM tb_voucher_order WHERE voucher_id=$VOUCHER;" -N 2>/dev/null)
echo "DB 订单数: $DB_CNT (应 == 成功数 $OK,允许短暂延迟差异)"

rm -rf "$TMP"
