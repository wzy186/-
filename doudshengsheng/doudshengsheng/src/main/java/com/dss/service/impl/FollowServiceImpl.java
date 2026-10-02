package com.dss.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.dto.Result;
import com.dss.entity.Follow;
import com.dss.mapper.FollowMapper;
import com.dss.service.IFollowService;
import com.dss.utils.RedisConstants;
import com.dss.utils.UserHolder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 关注 / 共同关注 / Feed 流(推模式)。
 * <p>
 * 关注关系:Redis Set(follow:{uid} -> Set(targetUid)) + DB 双写。
 * 共同关注:两个 Set 求交集。
 * Feed 流推模式:发笔记时遍历粉丝,把笔记 id 推到每个粉丝的 feed:{uid} ZSet(score=时间戳)。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FollowServiceImpl implements IFollowService {

    private final FollowMapper followMapper;
    private final StringRedisTemplate redis;

    @Override
    public Result follow(Long followUserId, Boolean isFollow) {
        Long userId = UserHolder.getUserId();
        if (userId == null) {
            return Result.fail(401, "未登录");
        }
        String key = RedisConstants.FOLLOW_KEY + userId;
        if (Boolean.TRUE.equals(isFollow)) {
            // 关注:写 DB + Redis Set
            Follow f = new Follow();
            f.setUserId(userId);
            f.setFollowUserId(followUserId);
            int rows = followMapper.insert(f);
            if (rows > 0) {
                redis.opsForSet().add(key, followUserId.toString());
            }
        } else {
            // 取关:删 DB + Redis Set
            followMapper.delete(new LambdaQueryWrapper<Follow>()
                    .eq(Follow::getUserId, userId)
                    .eq(Follow::getFollowUserId, followUserId));
            redis.opsForSet().remove(key, followUserId.toString());
        }
        return Result.ok();
    }

    @Override
    public Result isFollow(Long followUserId) {
        Long userId = UserHolder.getUserId();
        Boolean isMember = redis.opsForSet().isMember(RedisConstants.FOLLOW_KEY + userId, followUserId.toString());
        if (Boolean.TRUE.equals(isMember)) return Result.ok(true);
        // Redis 无记录(如 seed 关注没写 Redis),降级查 DB
        Long cnt = followMapper.selectCount(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Follow>()
                        .eq(Follow::getUserId, userId)
                        .eq(Follow::getFollowUserId, followUserId));
        return Result.ok(cnt > 0);
    }

    /**
     * 共同关注:当前用户与目标用户的关注 Set 求交集。
     */
    @Override
    public Result commonFollows(Long targetUserId) {
        Long userId = UserHolder.getUserId();
        // 优先 Redis Set 交集(关注操作时已写入)
        Set<String> intersect = redis.opsForSet().intersect(
                RedisConstants.FOLLOW_KEY + userId,
                RedisConstants.FOLLOW_KEY + targetUserId);
        // Redis 无数据时(如 seed 数据只写了 DB),降级查 DB 求交集
        if (intersect == null || intersect.isEmpty()) {
            List<Long> myFollows = new ArrayList<>(followMapper.selectList(
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Follow>()
                            .eq(Follow::getUserId, userId))
                    .stream().map(Follow::getFollowUserId).toList());
            List<Long> targetFollows = followMapper.selectList(
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Follow>()
                            .eq(Follow::getUserId, targetUserId))
                    .stream().map(Follow::getFollowUserId).toList();
            myFollows.retainAll(targetFollows); // 交集
            return Result.ok(myFollows);
        }
        List<Long> ids = intersect.stream().map(Long::valueOf).collect(Collectors.toList());
        return Result.ok(ids);
    }

    /**
     * 推模式 Feed:发笔记时,把笔记推到所有粉丝的收件箱 ZSet。
     */
    @Override
    public void pushToFollowers(Long blogId, Long authorId) {
        // 找粉丝(关注了 authorId 的人)。用 DB 查,followUserId=authorId 的 userId 列表
        List<Follow> fans = followMapper.selectList(
                new LambdaQueryWrapper<Follow>().eq(Follow::getFollowUserId, authorId));
        if (fans.isEmpty()) {
            return;
        }
        long now = System.currentTimeMillis();
        for (Follow fan : fans) {
            redis.opsForZSet().add(RedisConstants.FEED_KEY + fan.getUserId(), blogId.toString(), now);
        }
        log.debug("Feed 推送 blogId={} 到 {} 个粉丝", blogId, fans.size());
    }

    /**
     * 查收件箱:滚动分页。
     * max = 上一次最小 score(时间戳),offset = 与 max 相同 score 的个数。
     * 返回:笔记 id 列表 + minTime + offset,供下次滚动。
     */
    @Override
    public Result feed(Long max, Integer offset) {
        Long userId = UserHolder.getUserId();
        String key = RedisConstants.FEED_KEY + userId;
        if (max == null || max == 0) {
            max = System.currentTimeMillis();
        }
        if (offset == null) {
            offset = 0;
        }
        // ZREVRANGEBYSCORE key max min LIMIT offset count
        Set<org.springframework.data.redis.core.ZSetOperations.TypedTuple<String>> tuples =
                redis.opsForZSet().reverseRangeByScoreWithScores(key, 0, max, offset, 5);
        if (tuples == null || tuples.isEmpty()) {
            Map<String, Object> empty = new HashMap<>();
            empty.put("list", Collections.emptyList());
            empty.put("minTime", 0);
            empty.put("offset", 0);
            return Result.ok(empty);
        }
        List<Long> blogIds = new ArrayList<>();
        long minTime = Long.MAX_VALUE;
        int os = 0;
        for (org.springframework.data.redis.core.ZSetOperations.TypedTuple<String> t : tuples) {
            blogIds.add(Long.valueOf(t.getValue()));
            long time = t.getScore().longValue();
            if (time == minTime) {
                os++;
            } else if (time < minTime) {
                minTime = time;
                os = 1;
            }
        }
        Map<String, Object> resp = new HashMap<>();
        resp.put("list", blogIds);
        resp.put("minTime", minTime);
        resp.put("offset", os);
        return Result.ok(resp);
    }

    @Override
    public Result userProfile(Long userId) {
        // 关注数:我关注了多少人(user_id = userId)
        Long following = followMapper.selectCount(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Follow>()
                        .eq(Follow::getUserId, userId));
        // 粉丝数:多少人关注了我(follow_user_id = userId)
        Long followers = followMapper.selectCount(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Follow>()
                        .eq(Follow::getFollowUserId, userId));
        // 当前用户是否已关注目标
        Long currentUid = UserHolder.getUserId();
        boolean isFollowing = false;
        if (currentUid != null && !currentUid.equals(userId)) {
            Long cnt = followMapper.selectCount(
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Follow>()
                            .eq(Follow::getUserId, currentUid)
                            .eq(Follow::getFollowUserId, userId));
            isFollowing = cnt > 0;
        }
        Map<String, Object> data = new HashMap<>();
        data.put("userId", userId);
        data.put("following", following);
        data.put("followers", followers);
        data.put("isFollowing", isFollowing);
        data.put("isMe", currentUid != null && currentUid.equals(userId));
        return Result.ok(data);
    }
}
