package com.dss.annotation;

import java.lang.annotation.*;

/**
 * 标注该方法仅管理员可访问。非管理员抛 BizException(403)。
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface AdminOnly {
}
