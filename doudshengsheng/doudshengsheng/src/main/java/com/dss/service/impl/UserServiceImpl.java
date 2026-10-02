package com.dss.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.util.RandomUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.dto.LoginFormDTO;
import com.dss.dto.Result;
import com.dss.dto.UserDTO;
import com.dss.entity.User;
import com.dss.exception.BizException;
import com.dss.mapper.UserMapper;
import com.dss.service.IUserService;
import com.dss.utils.RegexUtils;
import com.dss.utils.RedisConstants;
import com.dss.utils.UserHolder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static com.dss.utils.RedisConstants.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserServiceImpl implements IUserService {

    private final UserMapper userMapper;
    private final StringRedisTemplate redis;

    @Override
    public Result sendCode(String phone) {
        // 手机号格式校验已在 Controller 层 @Pattern 完成
        // 1. 生成 6 位验证码
        String code = RandomUtil.randomNumbers(6);
        // 2. 存 Redis,2 分钟过期
        redis.opsForValue().set(LOGIN_CODE_KEY + phone, code, LOGIN_CODE_TTL, TimeUnit.SECONDS);
        // 3. 真实场景发短信,这里打印日志
        log.info("【兜省省】您的验证码:{},有效 2 分钟,请勿泄露。", code);
        return Result.ok();
    }

    @Override
    public Result login(LoginFormDTO form) {
        String phone = form.getPhone();
        // 1. 校验验证码
        String cacheCode = redis.opsForValue().get(LOGIN_CODE_KEY + phone);
        if (cacheCode == null || !cacheCode.equals(form.getCode())) {
            throw new BizException("验证码错误或已过期");
        }
        // 2. 查用户,不存在则注册
        User user = userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getPhone, phone));
        if (user == null) {
            user = createUserWithPhone(phone);
        }
        // 3. 生成 token,把 UserDTO 存进 Redis(Hash)
        String token = UUID.randomUUID().toString().replace("-", "");
        UserDTO userDTO = BeanUtil.copyProperties(user, UserDTO.class);
        Map<String, String> map = new HashMap<>();
        map.put("id", String.valueOf(userDTO.getId()));
        map.put("nickName", userDTO.getNickName() == null ? "" : userDTO.getNickName());
        map.put("icon", userDTO.getIcon() == null ? "" : userDTO.getIcon());
        map.put("role", String.valueOf(userDTO.getRole() == null ? 0 : userDTO.getRole()));
        redis.opsForHash().putAll(LOGIN_TOKEN_KEY + token, map);
        redis.expire(LOGIN_TOKEN_KEY + token, LOGIN_TOKEN_TTL, TimeUnit.SECONDS);

        return Result.ok(token);
    }

    @Override
    public User getById(Long id) {
        return userMapper.selectById(id);
    }

    @Override
    public UserDTO toDTO(User user) {
        return BeanUtil.copyProperties(user, UserDTO.class);
    }

    private User createUserWithPhone(String phone) {
        User user = new User();
        user.setPhone(phone);
        user.setNickName("兜省省用户_" + RandomUtil.randomString(6));
        userMapper.insert(user);
        return user;
    }
}
