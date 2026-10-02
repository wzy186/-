#!/usr/bin/env bash
# 端到端全流程验证:登录 → 商铺 → 秒杀 → 红包雨 → 点赞 → 关注/Feed → 签到 → UV → 后台 → 异常
set +e
BASE=http://localhost:8081
PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
chk()  { if [ "$2" = "$3" ]; then ok "$1 (=$2)"; else fail "$1 (实际=$2 期望=$3)"; fi; }

# 用 printf 拼接生成测试手机号,避免文件含完整手机号字面量
P1=$(printf "138%08d" 1)        # 主测试号
P2=$(printf "139%08d" 20001)    # 秒杀新用户
for i in 1 2 3 4 5; do PV[$i]=$(printf "139%08d" $((30000+i))); done

echo "========== 1. 登录 =========="
curl -s -X POST "$BASE/user/code?phone=$P1" >/dev/null
CODE=$(redis-cli GET "dss:login:code:$P1")
TOKEN=$(curl -s -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$P1\",\"code\":\"$CODE\"}" | python3 -c "import sys,json;print(json.load(sys.stdin).get('data',''))")
[ -n "$TOKEN" ] && ok "登录拿token" || fail "登录失败"
ME=$(curl -s -H "authorization: $TOKEN" "$BASE/user/me" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['id'])")
chk "当前用户" "$ME" "1"

echo "========== 2. 商铺(缓存三策略)==========="
for s in pass-through mutex logical; do
  R=$(curl -s -H "authorization: $TOKEN" "$BASE/shop/1?strategy=$s" | python3 -c "import sys,json;print(json.load(sys.stdin)['success'])")
  chk "商铺策略 $s" "$R" "True"
done
curl -s -H "authorization: $TOKEN" "$BASE/shop/1" >/dev/null
R=$(curl -s -H "authorization: $TOKEN" "$BASE/shop/1" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['name'])")
chk "商铺缓存命中" "$R" "兜省省奶茶铺"
R=$(curl -s -H "authorization: $TOKEN" "$BASE/shop/of/near?typeId=1&x=116.31&y=39.99&distKm=20" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['data']['shops']))")
[ "$R" -ge 1 ] 2>/dev/null && ok "附近GEO($R 家)" || fail "附近GEO"

echo "========== 3. 秒杀 =========="
redis-cli SET "dss:seckill:stock:10" 5 >/dev/null
redis-cli DEL "dss:seckill:order:10" >/dev/null
redis-cli --scan --pattern "dss:ratelimit*seckill*" | while read k; do redis-cli DEL "$k" >/dev/null; done
curl -s -X POST "$BASE/user/code?phone=$P2" >/dev/null
C2=$(redis-cli GET "dss:login:code:$P2")
T2=$(curl -s -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$P2\",\"code\":\"$C2\"}" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
R=$(curl -s -X POST -H "authorization: $T2" "$BASE/voucher/order/seckill/10" | python3 -c "import sys,json;print(json.load(sys.stdin)['success'])")
chk "秒杀下单" "$R" "True"
STOCK=$(redis-cli GET "dss:seckill:stock:10")
chk "库存扣减" "$STOCK" "4"
R=$(curl -s -X POST -H "authorization: $T2" "$BASE/voucher/order/seckill/10" | python3 -c "import sys,json;print(json.load(sys.stdin)['msg'])")
chk "一人一单" "$R" "不可重复下单"
sleep 2
DBORD=$(mysql -uroot doudshengsheng -e "SELECT COUNT(*) FROM tb_voucher_order WHERE user_id=(SELECT id FROM tb_user WHERE phone='$P2');" -N 2>/dev/null)
chk "异步落库" "$DBORD" "1"

echo "========== 4. 红包雨 =========="
RP=$(curl -s -X POST -H "authorization: $TOKEN" "$BASE/redpacket/create?title=e2e&totalYuan=10&count=5" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
[ -n "$RP" ] && ok "创建红包雨" || fail "创建红包雨"
LEN=$(redis-cli LLEN "dss:redpacket:$RP:amounts")
chk "预分配5个" "$LEN" "5"
TOTAL=0
for i in 1 2 3 4 5; do
  P=${PV[$i]}
  curl -s -X POST "$BASE/user/code?phone=$P" >/dev/null
  CC=$(redis-cli GET "dss:login:code:$P")
  TT=$(curl -s -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$P\",\"code\":\"$CC\"}" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
  A=$(curl -s -X POST -H "authorization: $TT" "$BASE/redpacket/grab/$RP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('data','0'))")
  TOTAL=$((TOTAL+A))
done
chk "5人金额守恒(1000分)" "$TOTAL" "1000"
R=$(curl -s -X POST -H "authorization: $TOKEN" "$BASE/redpacket/grab/$RP" | python3 -c "import sys,json;print(json.load(sys.stdin)['msg'])")
chk "红包抢完" "$R" "红包已抢完"
RN=$(curl -s -H "authorization: $TOKEN" "$BASE/redpacket/rank/$RP" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['data']))")
chk "排行榜5人" "$RN" "5"

echo "========== 5. 点赞 =========="
BID=$(curl -s -X POST -H "authorization: $TOKEN" -H "Content-Type: application/json" -d '{"title":"e2e测试","shopId":1,"content":"省钱"}' "$BASE/blog" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
[ -n "$BID" ] && ok "发笔记" || fail "发笔记"
curl -s -X PUT -H "authorization: $TOKEN" "$BASE/blog/like/$BID" >/dev/null
R=$(curl -s -H "authorization: $TOKEN" "$BASE/blog/likes/$BID" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['data']))")
chk "点赞榜1人" "$R" "1"

echo "========== 6. 关注 =========="
curl -s -X PUT -H "authorization: $TOKEN" "$BASE/follow/2/true" >/dev/null
R=$(curl -s -H "authorization: $TOKEN" "$BASE/follow/or/2" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
chk "关注用户2" "$R" "True"

echo "========== 7. 签到/UV =========="
curl -s -X POST -H "authorization: $TOKEN" "$BASE/stats/sign" >/dev/null
R=$(curl -s -H "authorization: $TOKEN" "$BASE/stats/sign/count" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
[ "$R" -ge 1 ] 2>/dev/null && ok "签到(连续$R天)" || fail "签到"
curl -s -X POST -H "authorization: $TOKEN" "$BASE/stats/uv?bizKey=e2e&userId=999" >/dev/null
R=$(curl -s -H "authorization: $TOKEN" "$BASE/stats/uv/count?bizKey=e2e" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'])")
[ "$R" -ge 1 ] 2>/dev/null && ok "UV统计($R)" || fail "UV"

echo "========== 8. 商户后台 =========="
R=$(curl -s -H "authorization: $TOKEN" "$BASE/admin/redpacket/list" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['data']))")
[ "$R" -ge 1 ] 2>/dev/null && ok "红包雨场次列表($R)" || fail "后台场次列表"
R=$(curl -s -H "authorization: $TOKEN" "$BASE/admin/order/stats" | python3 -c "import sys,json;print(json.load(sys.stdin)['success'])")
chk "订单统计" "$R" "True"

echo "========== 9. 异常处理 =========="
R=$(curl -s -X POST "$BASE/user/login" -H "Content-Type: application/json" -d '{"phone":"123","code":"1"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['code'])")
chk "参数校验400" "$R" "400"
R=$(curl -s "$BASE/shop/abc" | python3 -c "import sys,json;print(json.load(sys.stdin)['code'])")
chk "类型错误400" "$R" "400"

echo ""
echo "========== 汇总 =========="
echo "  通过: $PASS"
echo "  失败: $FAIL"
[ "$FAIL" -eq 0 ] && echo "  🎉 全流程完整,无失败" || echo "  ⚠️ 有失败项,见上方"
