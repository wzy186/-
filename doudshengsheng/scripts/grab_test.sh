#!/usr/bin/env bash
# 多用户抢红包压测脚本
RP=${1:-353831436047876097}
BASE=http://localhost:8081
echo "红包场次: $RP"
echo "用户          抢到(分)  状态"
for phone in 13800000002 13800000003 13800000004 13800000005 13800000006 13800000007 13800000008 13800000009 13800000010; do
  # 发码
  curl -s -X POST "$BASE/user/code?phone=$phone" >/dev/null
  # 直接从 Redis 拿验证码(比解析日志可靠)
  code=$(redis-cli GET "dss:login:code:$phone")
  # 登录
  tk=$(curl -s -X POST "$BASE/user/login" -H "Content-Type: application/json" -d "{\"phone\":\"$phone\",\"code\":\"$code\"}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data',''))")
  if [ -z "$tk" ]; then echo "$phone  登录失败"; continue; fi
  # 抢红包
  resp=$(curl -s -X POST -H "authorization: $tk" "$BASE/redpacket/grab/$RP")
  amt=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data','-'))")
  echo "$phone  $amt"
done
echo "=== 排行榜 ==="
curl -s "$BASE/redpacket/rank/$RP" | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
for i,r in enumerate(d,1):
    print(f'  第{i}名  用户{r[\"userId\"]}  {r[\"amount\"]}分')
print(f'共 {len(d)} 人领取')
"
echo "=== meta ==="; redis-cli HGETALL "dss:redpacket:$RP:meta" | paste - - | tr '\t' ' '
