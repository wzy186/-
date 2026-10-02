package com.dss.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.dss.dto.Result;
import com.dss.dto.UserDTO;
import com.dss.entity.Blog;
import com.dss.entity.User;
import com.dss.mapper.BlogMapper;
import com.dss.mapper.UserMapper;
import com.dss.service.IBlogService;
import com.dss.service.IFollowService;
import com.dss.utils.RedisConstants;
import com.dss.utils.UserHolder;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ZSetOperations;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * 探店笔记:热门分页 + 点赞 + 点赞排行榜(ZSet 按时间戳)。
 */
@Service
@RequiredArgsConstructor
public class BlogServiceImpl implements IBlogService {

    private final BlogMapper blogMapper;
    private final UserMapper userMapper;
    private final StringRedisTemplate redis;
    private final IFollowService followService;

    @Override
    public Result queryHotBlog(Integer current) {
        Page<Blog> page = blogMapper.selectPage(new Page<>(current == null ? 1 : current, 5),
                new LambdaQueryWrapper<Blog>().orderByDesc(Blog::getLiked));
        // 附带作者昵称,前端列表直接显示
        List<Map<String, Object>> list = new ArrayList<>();
        for (Blog b : page.getRecords()) {
            Map<String, Object> m = new HashMap<>();
            m.put("id", b.getId());
            m.put("title", b.getTitle());
            m.put("content", b.getContent());
            m.put("shopId", b.getShopId());
            m.put("liked", b.getLiked());
            m.put("userId", b.getUserId());
            User author = userMapper.selectById(b.getUserId());
            m.put("authorName", author == null ? "匿名" : author.getNickName());
            list.add(m);
        }
        return Result.ok(list);
    }

    @Override
    public Result queryBlogById(Long id) {
        Blog blog = blogMapper.selectById(id);
        if (blog == null) {
            return Result.fail("笔记不存在");
        }
        // 拼作者信息 + 当前用户是否已赞
        Map<String, Object> data = new HashMap<>();
        data.put("id", blog.getId());
        data.put("title", blog.getTitle());
        data.put("content", blog.getContent());
        data.put("shopId", blog.getShopId());
        data.put("images", blog.getImages());
        data.put("liked", blog.getLiked());
        data.put("comments", blog.getComments());
        data.put("createTime", blog.getCreateTime());
        // 作者
        User author = userMapper.selectById(blog.getUserId());
        if (author != null) {
            UserDTO authorDTO = new UserDTO();
            authorDTO.setId(author.getId());
            authorDTO.setNickName(author.getNickName());
            authorDTO.setIcon(author.getIcon());
            data.put("author", authorDTO);
            data.put("userId", author.getId());
        }
        // 当前用户是否已赞
        Long currentUid = UserHolder.getUserId();
        if (currentUid != null) {
            Double score = redis.opsForZSet().score(RedisConstants.BLOG_LIKED_KEY + id, currentUid.toString());
            data.put("isLike", score != null);
        } else {
            data.put("isLike", false);
        }
        return Result.ok(data);
    }

    @Override
    public Result queryBlogByUserId(Long userId) {
        List<Blog> blogs = blogMapper.selectList(
                new LambdaQueryWrapper<Blog>().eq(Blog::getUserId, userId).orderByDesc(Blog::getCreateTime));
        return Result.ok(blogs);
    }

    /**
     * 点赞 / 取消点赞:ZSet 存(uid, 时间戳)实现一人一赞 + 排行榜。
     */
    @Override
    public Result likeBlog(Long id) {
        Long userId = UserHolder.getUserId();
        if (userId == null) {
            return Result.fail(401, "未登录");
        }
        String key = RedisConstants.BLOG_LIKED_KEY + id;
        Double score = redis.opsForZSet().score(key, userId.toString());
        if (score == null) {
            // 未点赞 → 点赞(liked + 1)
            blogMapper.update(null, new LambdaUpdateWrapper<Blog>()
                    .eq(Blog::getId, id).setSql("liked = liked + 1"));
            redis.opsForZSet().add(key, userId.toString(), System.currentTimeMillis());
        } else {
            // 已点赞 → 取消(liked - 1)
            blogMapper.update(null, new LambdaUpdateWrapper<Blog>()
                    .eq(Blog::getId, id).setSql("liked = liked - 1"));
            redis.opsForZSet().remove(key, userId.toString());
        }
        return Result.ok();
    }

    /**
     * 点赞排行榜:Top5 最早点赞的人(ZSet 按时间戳升序)
     */
    @Override
    public Result queryBlogLikes(Long id) {
        String key = RedisConstants.BLOG_LIKED_KEY + id;
        Set<ZSetOperations.TypedTuple<String>> top = redis.opsForZSet()
                .rangeWithScores(key, 0, 4);
        if (top == null || top.isEmpty()) {
            return Result.ok(Collections.emptyList());
        }
        List<Map<String, Object>> result = new ArrayList<>();
        for (ZSetOperations.TypedTuple<String> t : top) {
            Map<String, Object> m = new HashMap<>();
            m.put("userId", t.getValue());
            m.put("likeTime", t.getScore().longValue());
            result.add(m);
        }
        return Result.ok(result);
    }

    @Override
    public Result saveBlog(Blog blog) {
        Long userId = UserHolder.getUserId();
        blog.setUserId(userId);
        blog.setLiked(0);
        blogMapper.insert(blog);
        // 推模式 Feed:把笔记推送到粉丝收件箱
        followService.pushToFollowers(blog.getId(), userId);
        return Result.ok(blog.getId());
    }
}
