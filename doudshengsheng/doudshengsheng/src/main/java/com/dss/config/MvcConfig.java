package com.dss.config;

import com.dss.interceptor.LoginInterceptor;
import com.dss.interceptor.RefreshTokenInterceptor;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class MvcConfig implements WebMvcConfigurer {

    private final StringRedisTemplate redis;

    public MvcConfig(StringRedisTemplate redis) {
        this.redis = redis;
    }

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        // 拦截器一:拦截全部,只刷新 token
        registry.addInterceptor(new RefreshTokenInterceptor(redis))
                .addPathPatterns("/**")
                .order(0);
        // 拦截器二:需要登录的路径(/user/code、/user/login 排除)
        registry.addInterceptor(new LoginInterceptor())
                .addPathPatterns(
                        "/user/me",
                        "/shop/type/**",
                        "/voucher/order/**",
                        "/blog/**",
                        "/follow/**",
                        "/stats/**",
                        "/redpacket/**",
                        "/admin/**",
                        "/ai/**"
                )
                .order(1);
    }
}
