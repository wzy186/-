package com.dss.aspect;

import com.dss.annotation.AdminOnly;
import com.dss.dto.UserDTO;
import com.dss.exception.BizException;
import com.dss.utils.UserHolder;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

/**
 * 管理员权限校验切面:@AdminOnly 方法只允许 role=1 的用户访问。
 */
@Aspect
@Component
public class AdminAuthAspect {

    @Around("@annotation(adminOnly)")
    public Object check(ProceedingJoinPoint pjp, AdminOnly adminOnly) throws Throwable {
        UserDTO user = UserHolder.getUser();
        if (user == null) {
            throw new BizException(401, "未登录");
        }
        if (user.getRole() == null || user.getRole() != 1) {
            throw new BizException(403, "无权限,仅管理员可访问");
        }
        return pjp.proceed();
    }
}
