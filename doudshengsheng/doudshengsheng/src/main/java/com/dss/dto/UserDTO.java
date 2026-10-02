package com.dss.dto;

import lombok.Data;

/**
 * 脱敏后的用户信息,放 ThreadLocal 和 Token
 */
@Data
public class UserDTO {
    private Long id;
    private String nickName;
    private String icon;
    private Integer role; // 0普通 1管理员
}
