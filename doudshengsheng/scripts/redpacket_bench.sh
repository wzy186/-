#!/usr/bin/env bash
# 红包雨并发压测 + 金额守恒验证
# 用法:bash redpacket_bench.sh [用户数] [总金额元] [红包个数]
# 流程:创建红包 → N 用户并发抢 → 统计 + 验证金额守恒
set -e

USERS=${1:-300}
TOTAL_YUAN=${2:-100}
COUNT=${3:-50}
BASE=http://localhost:8081

echo "=== 红包雨压测: $USERS 用户抢红包(总 $TOTAL_YUAN 元 / $COUNT 个)==="

# 0. 登录管理员(创建红包用)+ 准备用户
ADMIN_PHONE=13800000001
curl -s -X POST "$BASE/user/code?phone=$ADMIN_PHONE" >/dev/null
code=$(redis-cli GET "dss:login:code:$ADMIN_PHONE")
ADMIN_TK=$(curl -s -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$ADMIN_PHONE\",\"code\":\"$code\"}" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")

echo "准备 $USERS 个抢红包用户 token..."
TOKENS=()
for i in $(seq 1 $USERS); do
  phone=$(printf "139%08d" $((20000 + i)))
  curl -s --max-time 5 -X POST "$BASE/user/code?phone=$phone" >/dev/null
  c=$(redis-cli GET "dss:login:code:$phone")
  tk=$(curl -s --max-time 5 -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$phone\",\"code\":\"$c\"}" | python3 -c "import sys,json;print(json.load(sys.stdin).get('data',''))")
  TOKENS+=("$tk")
  if [ $((i % 50)) -eq 0 ]; then echo "  已准备 $i / $USERS"; fi
done

# 1. 创建红包
echo "创建红包: $TOTAL_YUAN 元 / $COUNT 个..."
RP=$(curl -s -X POST -H "authorization: $ADMIN_TK" "$BASE/redpacket/create?title=bench&totalYuan=$TOTAL_YUAN&count=$COUNT" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
echo "红包场次 ID: $RP"

# 2. 并发抢
echo "开始并发抢红包..."
TMP=$(mktemp -d)
START=$(python3 -c 'import time;print(time.time())')
for tk in "${TOKENS[@]}"; do
  (
    resp=$(curl -s --max-time 5 -X POST -H "authorization: $tk" "$BASE/redpacket/grab/$RP")
    amt=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data',''))" 2>/dev/null)
    msg=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('msg','')[:16])" 2>/dev/null)
    if [ -n "$amt" ] && [ "$amt" != "None" ] && [ "$amt" != "" ]; then
      echo "OK|$amt" >> "$TMP/results.txt"
    else
      echo "FAIL|$msg" >> "$TMP/results.txt"
    fi
  ) &
done
wait
END=$(python3 -c 'import time;print(time.time())')

# 3. 统计
OK=$(grep -c '^OK|' "$TMP/results.txt")
FAIL=$(grep -c '^FAIL|' "$TMP/results.txt")
COST=$(python3 -c "print(round($END-$START, 3))")
# 抢到的金额求和(分)
GOT_SUM=$(grep '^OK|' "$TMP/results.txt" | cut -d'|' -f2 | python3 -c "import sys;print(sum(int(x) for x in sys.stdin if x.strip()))")
TOTAL_CENTS=$((TOTAL_YUAN * 100))

echo ""
echo "=== 结果 ==="
echo "总用户: $USERS"
echo "红包个数: $COUNT"
echo "抢到人数: $OK"
echo "失败人数: $FAIL"
echo "耗时: ${COST}s"
echo "失败原因分布:"
grep '^FAIL|' "$TMP/results.txt" | cut -d'|' -f2 | sort | uniq -c | sort -rn | head -5

# 4. 守恒验证
echo ""
echo "=== 金额守恒验证 ==="
echo "红包总额: $TOTAL_CENTS 分 ($TOTAL_YUAN 元)"
echo "抢到金额之和: $GOT_SUM 分 ($(python3 -c "print(round($GOT_SUM/100,2))") 元)"
if [ "$GOT_SUM" = "$TOTAL_CENTS" ]; then
  echo "✅ 金额守恒: 抢到的之和 == 红包总额"
else
  echo "⚠️ 金额不守恒: 差额 $((TOTAL_CENTS - GOT_SUM)) 分(可能有红包未抢完)"
fi

# 5. 无超领验证
echo ""
echo "=== 无超领验证 ==="
if [ "$OK" -le "$COUNT" ]; then
  echo "✅ 无超领: 抢到人数($OK) <= 红包个数($COUNT)"
else
  echo "❌ 超领: 抢到人数($OK) > 红包个数($COUNT)"
fi

# 6. Redis meta 对比
echo ""
echo "=== Redis meta ==="
redis-cli HGETALL "dss:redpacket:$RP:meta" | paste - - | tr '\t' ' '

rm -rf "$TMP"
