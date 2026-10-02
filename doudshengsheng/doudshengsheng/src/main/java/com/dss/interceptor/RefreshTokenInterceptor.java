package com.dss.interceptor;

import cn.hutool.core.bean.BeanUtil;
import com.dss.dto.UserDTO;
import com.dss.utils.RedisConstants;
import com.dss.utils.UserHolder;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.web.servlet.HandlerInterceptor;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 拦截器一:刷新 Token 有效期 + 把用户信息放入 ThreadLocal。
 * 拦截所有路径,只做"读取并续期",不做拦截。
 */
public class RefreshTokenInterceptor implements HandlerInterceptor {

    private final StringRedisTemplate redis;

    public RefreshTokenInterceptor(StringRedisTemplate redis) {
        this.redis = redis;
    }

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        // 1. 从 header 取 token
        String token = request.getHeader("authorization");
        if (token == null || token.isEmpty()) {
            return true; // 没 token 也放行,交给登录拦截器决定
        }
        // 2. 取用户
        Map<Object, Object> userMap = redis.opsForHash().entries(RedisConstants.LOGIN_TOKEN_KEY + token);
        if (userMap.isEmpty()) {
            return true;
        }
        // 3. 转 DTO 放 ThreadLocal
        UserDTO userDTO = BeanUtil.fillBeanWithMap(userMap, new UserDTO(), false);
        UserHolder.saveUser(userDTO);
        // 4. 刷新 token 有效期
        redis.expire(RedisConstants.LOGIN_TOKEN_KEY + token, RedisConstants.LOGIN_TOKEN_TTL, TimeUnit.SECONDS);
        return true;
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) {
        UserHolder.removeUser();
    }
}
