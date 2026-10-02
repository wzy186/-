package com.dss.utils;

import java.util.regex.Pattern;

/**
 * 手机号校验(脱敏:仅校验 11 位数字 + 1 开头,不固化具体号段)
 */
public class RegexUtils {

    private static final Pattern PHONE = Pattern.compile("^1\\d{10}$");

    public static boolean isPhoneInvalid(String phone) {
        return phone == null || !PHONE.matcher(phone).matches();
    }

    private RegexUtils() {}
}
