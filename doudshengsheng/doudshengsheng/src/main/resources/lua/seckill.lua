-- 秒杀下单 Lua 脚本:判断时间 + 判断库存 + 判断一人一单 + 扣库存 + 记录用户,全部原子
-- KEYS[1] = seckill:stock:{voucherId}    库存(字符串,用 DECR)
-- KEYS[2] = seckill:order:{voucherId}    已购用户 Set
-- ARGV[1] = userId
-- ARGV[2] = 当前时间戳(毫秒)
-- ARGV[3] = 开始时间(毫秒)
-- ARGV[4] = 结束时间(毫秒)
-- 返回:0 成功;1 未开始;2 已结束;3 库存不足;4 重复下单

local voucherId = KEYS[1]
local userId = ARGV[1]
local now = tonumber(ARGV[2])
local beginTime = tonumber(ARGV[3])
local endTime = tonumber(ARGV[4])

-- 1. 时间判断
if now < beginTime then
    return 1
end
if now > endTime then
    return 2
end

-- 2. 判断是否已下过单(SISMEMBER)
local isMember = redis.call('SISMEMBER', KEYS[2], userId)
if isMember == 1 then
    return 4
end

-- 3. 库存判断
local stock = tonumber(redis.call('GET', KEYS[1]))
if stock == nil or stock <= 0 then
    return 3
end

-- 4. 扣库存 + 记录用户
redis.call('DECR', KEYS[1])
redis.call('SADD', KEYS[2], userId)
return 0
