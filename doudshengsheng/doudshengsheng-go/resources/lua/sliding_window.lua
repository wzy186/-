-- 滑动窗口限流(按 ZSet + 时间戳):用于红包雨防刷
-- KEYS[1] = redpacket:rate:{redpacketId}:{uid}
-- ARGV[1] = 当前时间戳(毫秒)
-- ARGV[2] = 窗口大小(毫秒)
-- ARGV[3] = 窗口内最大次数
-- ARGV[4] = 唯一成员标识(用毫秒+随机避免覆盖,这里用时间戳)
-- 返回:1 放行;0 拒绝

local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local maxCount = tonumber(ARGV[3])
local member = ARGV[4]

-- 1. 清除窗口外的旧记录
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)

-- 2. 统计当前窗口内次数
local count = redis.call('ZCARD', KEYS[1])
if count >= maxCount then
    return 0
end

-- 3. 记录本次
redis.call('ZADD', KEYS[1], now, member)
redis.call('PEXPIRE', KEYS[1], window)
return 1
