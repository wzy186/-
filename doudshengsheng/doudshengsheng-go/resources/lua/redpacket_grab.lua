-- 红包雨抢红包 Lua 脚本:幂等判断 + 取金额 + 记录领取 + 减余量,原子执行
-- KEYS[1] = redpacket:{id}:amounts   金额 List(RPOP 取尾)
-- KEYS[2] = redpacket:{id}:taken      已领 Hash,field=uid,value=amount
-- KEYS[3] = redpacket:{id}:meta       元数据 Hash
-- ARGV[1] = userId
-- ARGV[2] = 当前时间戳(秒,记录领取时间)
-- 返回:
--   "金额(分)"  抢到
--   "-1"        已领过(幂等)
--   "0"         红包已抢完

local uid = ARGV[1]

-- 1. 幂等:已领过直接返回 -1
if redis.call('HEXISTS', KEYS[2], uid) == 1 then
    return '-1'
end

-- 2. 取金额(RPOP)
local amount = redis.call('RPOP', KEYS[1])
if not amount then
    return '0'
end

-- 3. 记录领取(uid -> amount)
redis.call('HSET', KEYS[2], uid, amount)
redis.call('HSET', KEYS[2], uid .. ':time', ARGV[2])

-- 4. 减余量
local remain = redis.call('HINCRBY', KEYS[3], 'remain', -1)
redis.call('HINCRBY', KEYS[3], 'got', 1)

-- 5. 抢完则置状态=2,并设置到期清理标记
if tonumber(remain) == 0 then
    redis.call('HSET', KEYS[3], 'status', '2')
end

return amount
