package com.dss.utils;

import com.dss.dto.UserDTO;

/**
 * ThreadLocal 用户上下文,保存当前登录用户
 */
public class UserHolder {

    private static final ThreadLocal<UserDTO> TL = new ThreadLocal<>();

    public static void saveUser(UserDTO user) {
        TL.set(user);
    }

    public static UserDTO getUser() {
        return TL.get();
    }

    public static Long getUserId() {
        UserDTO user = TL.get();
        return user == null ? null : user.getId();
    }

    public static void removeUser() {
        TL.remove();
    }
}
